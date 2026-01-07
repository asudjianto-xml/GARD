"""
GARD v2: Two-Gate Decision Rule (Conflict vs Weakness Separation)

Implements:
1. Conflict gate: C >= C_hi → abstain_conflict
2. Weakness gate: C < C_hi and GISR < 1 → abstain_weak
3. Accept: otherwise

Risk score for ranking: λ*C + (1-λ)*σ((1-GISR)/t)
"""

import torch
from typing import Dict, List, Tuple
import numpy as np


def compute_coherence_v2(V: torch.Tensor, eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    """
    Compute bivector coherence using pairwise dot products.

    Exact implementation per GARD v2 specification:
    B² = sum_{i<j} (1 - (v_i^T v_j)²)
    D = n(n-1)/2
    C = sqrt(B²/D)

    Args:
        V: Evidence embeddings (B, n, d) or (n, d), must be normalized
        eps: Small epsilon for numerical stability

    Returns:
        Dictionary with:
        - C: Coherence ratio (B,) or scalar, in [0, 1]
        - B2: Bivector squared norm (B,) or scalar
        - D: Number of pairs (B,) or scalar
    """
    # Handle single batch case
    if V.dim() == 2:
        V = V.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, n, d = V.shape

    # Handle edge cases
    if n < 2:
        C = torch.zeros(B, device=V.device, dtype=V.dtype)
        B2 = torch.zeros(B, device=V.device, dtype=V.dtype)
        D = torch.zeros(B, device=V.device, dtype=V.dtype)
        result = {"C": C, "B2": B2, "D": D}
        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    # Compute Gram matrix: S = V @ V^T  (B, n, n)
    S = torch.bmm(V, V.transpose(1, 2))  # (B, n, n)

    # Extract upper triangle without diagonal
    # Create mask for upper triangle
    mask = torch.triu(torch.ones(n, n, device=V.device, dtype=torch.bool), diagonal=1)

    # Compute 1 - S² for upper triangle
    S_sq = S * S  # (B, n, n)
    one_minus_S_sq = 1.0 - S_sq  # (B, n, n)

    # Sum over upper triangle
    B2 = torch.zeros(B, device=V.device, dtype=V.dtype)
    for b in range(B):
        B2[b] = one_minus_S_sq[b][mask].sum()

    # Denominator: number of pairs
    D = torch.full((B,), n * (n - 1) / 2.0, device=V.device, dtype=V.dtype)

    # Coherence ratio
    C = torch.where(D > eps, torch.sqrt(B2 / D + eps), torch.zeros_like(D))
    C = torch.clamp(C, 0.0, 1.0)

    result = {
        "C": C,
        "B2": B2,
        "D": D,
    }

    if squeeze:
        result = {k: v.squeeze(0) if v.dim() > 0 else v for k, v in result.items()}

    return result


def compute_margin_v2(V: torch.Tensor, q: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute geometric margin.

    M = sum_i v_i^T q

    Args:
        V: Evidence embeddings (B, n, d) or (n, d), normalized
        q: Query embedding (B, d) or (d), normalized

    Returns:
        Dictionary with:
        - M: Margin (B,) or scalar
        - absM: Absolute margin (B,) or scalar
        - s: Individual alignments (B, n) or (n,)
    """
    # Handle single batch case
    if V.dim() == 2:
        V = V.unsqueeze(0)
        q = q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Compute alignments: s_i = v_i^T q
    s = torch.einsum("bnd,bd->bn", V, q)  # (B, n)

    # Margin
    M = s.sum(dim=1)  # (B,)
    absM = M.abs()

    result = {
        "M": M,
        "absM": absM,
        "s": s,
    }

    if squeeze:
        result = {k: v.squeeze(0) if v.dim() > 0 else v for k, v in result.items()}

    return result


def compute_gisr_v2(
    V: torch.Tensor,
    q: torch.Tensor,
    eta: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute GISR (Geometric Insufficiency Sufficiency Ratio).

    GISR = |M| / (n * η)

    Note: L_Q = 1 for normalized vectors.

    Args:
        V: Evidence embeddings (B, n, d) or (n, d), normalized
        q: Query embedding (B, d) or (d), normalized
        eta: Perturbation radius
        eps: Small epsilon

    Returns:
        Dictionary with:
        - gisr: GISR value (B,) or scalar
        - tau: Threshold (B,) or scalar
    """
    # Handle single batch case
    if V.dim() == 2:
        V = V.unsqueeze(0)
        q = q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    n = V.shape[1]

    # Compute margin
    margin_dict = compute_margin_v2(V, q)
    absM = margin_dict["absM"]

    # Threshold (L_Q = 1 for normalized)
    tau = torch.full((V.shape[0],), float(n * eta), device=V.device, dtype=V.dtype)

    # GISR
    gisr = absM / (tau + eps)  # (B,)

    result = {
        "gisr": gisr,
        "tau": tau,
    }

    if squeeze:
        result = {k: v.squeeze(0) for k, v in result.items()}

    return result


def compute_risk_score_v2(
    C: torch.Tensor,
    gisr: torch.Tensor,
    lam: float = 0.7,
    t: float = 0.25,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute GARD v2 risk score for ranking.

    Risk = λ * C + (1-λ) * σ((1-GISR)/t)

    where σ(z) = 1/(1 + e^(-z))

    Args:
        C: Coherence (B,) or scalar
        gisr: GISR value (B,) or scalar
        lam: Weight for conflict term (default 0.7)
        t: Temperature for sigmoid (default 0.25)
        eps: Small epsilon

    Returns:
        Risk score (B,) or scalar
    """
    # Sigmoid term for weakness
    z = (1.0 - gisr) / (t + eps)
    sigma_z = torch.sigmoid(z)

    # Combined risk
    risk = lam * C + (1 - lam) * sigma_z

    return risk


def decide_v2(
    C: float,
    gisr: float,
    C_hi: float = 0.35,
) -> Tuple[str, bool]:
    """
    GARD v2 two-gate decision rule.

    Gate 1 (Conflict): If C >= C_hi → abstain_conflict
    Gate 2 (Weakness): If C < C_hi and GISR < 1 → abstain_weak
    Otherwise: accept

    Args:
        C: Coherence value
        gisr: GISR value
        C_hi: Conflict threshold (default 0.35)

    Returns:
        Tuple of (decision, abstain_flag)
        - decision: "accept", "abstain_conflict", or "abstain_weak"
        - abstain_flag: True if any abstain, False if accept
    """
    # Gate 1: Conflict detection
    if C >= C_hi:
        return "abstain_conflict", True

    # Gate 2: Weakness detection
    if gisr < 1.0:
        return "abstain_weak", True

    # Accept
    return "accept", False


class GARDv2:
    """
    GARD v2: Two-Gate Decision Rule.

    Separates conflict detection (hard abstain) from weakness detection (soft abstain).

    Args:
        eta: Perturbation radius (default 0.1)
        C_hi: Conflict threshold (default 0.35)
        lam: Risk score weight for conflict term (default 0.7)
        t: Temperature for risk score sigmoid (default 0.25)
        device: torch device (default "cuda")
        eps: Small epsilon for numerical stability (default 1e-12)
    """

    def __init__(
        self,
        eta: float = 0.1,
        C_hi: float = 0.35,
        lam: float = 0.7,
        t: float = 0.25,
        device: str = "cuda",
        eps: float = 1e-12,
    ):
        self.eta = eta
        self.C_hi = C_hi
        self.lam = lam
        self.t = t
        self.device = device
        self.eps = eps

    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings to unit length."""
        norms = torch.linalg.norm(embeddings, dim=-1, keepdim=True)
        return embeddings / (norms + self.eps)

    def score_single(
        self,
        V: torch.Tensor,
        q: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Score a single example.

        Args:
            V: Evidence embeddings (n, d), unnormalized
            q: Query embedding (d,), unnormalized

        Returns:
            Dictionary with:
            - C: Coherence
            - M: Margin
            - absM: Absolute margin
            - gisr: GISR
            - risk_score: Combined risk score
            - decision: "accept", "abstain_conflict", or "abstain_weak"
            - abstain: Boolean
        """
        # Normalize
        V_norm = self.normalize_embeddings(V)
        q_norm = self.normalize_embeddings(q)

        # Compute metrics
        coherence_dict = compute_coherence_v2(V_norm, self.eps)
        margin_dict = compute_margin_v2(V_norm, q_norm)
        gisr_dict = compute_gisr_v2(V_norm, q_norm, self.eta, self.eps)

        C = coherence_dict["C"].item()
        M = margin_dict["M"].item()
        absM = margin_dict["absM"].item()
        gisr = gisr_dict["gisr"].item()

        # Compute risk score
        risk_score = compute_risk_score_v2(
            coherence_dict["C"],
            gisr_dict["gisr"],
            self.lam,
            self.t,
            self.eps,
        ).item()

        # Make decision
        decision, abstain = decide_v2(C, gisr, self.C_hi)

        return {
            "C": C,
            "M": M,
            "absM": absM,
            "gisr": gisr,
            "risk_score": risk_score,
            "decision": decision,
            "abstain": abstain,
        }

    def batch_score(
        self,
        V_batch: List[torch.Tensor],
        q_batch: List[torch.Tensor],
    ) -> List[Dict[str, float]]:
        """
        Score a batch of examples.

        Args:
            V_batch: List of evidence embeddings (n_i, d)
            q_batch: List of query embeddings (d,)

        Returns:
            List of score dictionaries
        """
        results = []
        for V, q in zip(V_batch, q_batch):
            results.append(self.score_single(V, q))
        return results
