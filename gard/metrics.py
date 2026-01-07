"""
GARD metrics: Margin, Coherence, GISR, Risk.

Exact implementation per Section 4 of instruction pack.
"""

import torch
from typing import Dict


def compute_margin(V: torch.Tensor, q: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute geometric margin.

    M = sum_i v_i^T q

    Args:
        V: Evidence embeddings (B, n, d) or (n, d)
        q: Query embedding (B, d) or (d)

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


def compute_coherence(V: torch.Tensor, eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    """
    Compute bivector coherence using O(n) prefix method.

    Exact implementation per Section 4.2 specification.

    Args:
        V: Evidence embeddings (B, n, d) or (n, d)
        eps: Small epsilon for numerical stability

    Returns:
        Dictionary with:
        - C: Coherence ratio (B,) or scalar, clamped to [0, 1]
        - B_norm: Bivector norm (B,) or scalar
        - D: Denominator (B,) or scalar
    """
    # Handle single batch case
    if V.dim() == 2:
        V = V.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Prefix method (O(n))
    P = torch.cumsum(V, dim=1) - V  # (B, n, d)
    A = torch.einsum("bnd,bne->bde", P, V)  # (B, d, d)
    W = A - A.transpose(-1, -2)  # (B, d, d)

    # Bivector norm
    W_fro2 = (W * W).sum(dim=(-1, -2))  # (B,)
    B_norm = torch.sqrt(0.5 * W_fro2 + eps)  # (B,)

    # Denominator
    r = torch.linalg.norm(V, dim=-1)  # (B, n)
    sum_r = r.sum(dim=1)  # (B,)
    sum_r2 = (r * r).sum(dim=1)  # (B,)
    D = 0.5 * (sum_r * sum_r - sum_r2)  # (B,)

    # Coherence ratio
    C = torch.where(D > eps, B_norm / D, torch.zeros_like(D))
    C = torch.clamp(C, 0.0, 1.0)

    result = {
        "C": C,
        "B_norm": B_norm,
        "D": D,
    }

    if squeeze:
        result = {k: v.squeeze(0) if v.dim() > 0 else v for k, v in result.items()}

    return result


def compute_gisr(
    V: torch.Tensor,
    q: torch.Tensor,
    eta: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute GISR (Geometric Insufficiency Sufficiency Ratio).

    GISR = |M| / (n * L_Q * eta)

    Args:
        V: Evidence embeddings (B, n, d) or (n, d)
        q: Query embedding (B, d) or (d)
        eta: Perturbation radius
        eps: Small epsilon

    Returns:
        Dictionary with:
        - gisr: GISR value (B,) or scalar
        - tau: Threshold (B,) or scalar
        - L_Q: Lipschitz constant (B,) or scalar
        - decision_accept: Boolean (B,) or scalar
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
    margin_dict = compute_margin(V, q)
    absM = margin_dict["absM"]

    # Lipschitz constant (query norm)
    L_Q = torch.linalg.norm(q, dim=-1)  # (B,)

    # Threshold
    tau = n * L_Q * eta  # (B,)

    # GISR
    gisr = absM / (tau + eps)  # (B,)

    # Decision: accept if GISR >= 1
    decision_accept = gisr >= 1.0

    result = {
        "gisr": gisr,
        "tau": tau,
        "L_Q": L_Q,
        "decision_accept": decision_accept,
    }

    if squeeze:
        result = {k: v.squeeze(0) if v.dim() > 0 else v for k, v in result.items()}

    return result


def compute_risk_ga(
    V: torch.Tensor,
    q: torch.Tensor,
    eta: float,
    alpha: float = 0.5,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute combined GA risk score.

    risk_GA = alpha * (1 / (GISR + eps)) + (1 - alpha) * C

    Args:
        V: Evidence embeddings (B, n, d) or (n, d)
        q: Query embedding (B, d) or (d)
        eta: Perturbation radius
        alpha: Weight for GISR term (default 0.5)
        eps: Small epsilon

    Returns:
        Dictionary with all metrics including:
        - risk_ga: Combined risk score (B,) or scalar
    """
    # Compute GISR
    gisr_dict = compute_gisr(V, q, eta, eps)

    # Compute coherence
    coherence_dict = compute_coherence(V, eps)

    # Compute margin
    margin_dict = compute_margin(V, q)

    # Combined risk
    risk_ga = alpha * (1.0 / (gisr_dict["gisr"] + eps)) + (1 - alpha) * coherence_dict["C"]

    # Combine all results
    result = {
        **margin_dict,
        **coherence_dict,
        **gisr_dict,
        "risk_ga": risk_ga,
    }

    return result


def compute_all_gard_metrics(
    V: torch.Tensor,
    q: torch.Tensor,
    eta: float,
    alpha: float = 0.5,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Compute all GARD metrics in one call.

    Convenience function that computes:
    - Margin (M, absM, s)
    - Coherence (C, B_norm, D)
    - GISR (gisr, tau, L_Q, decision_accept)
    - Risk (risk_ga)

    Args:
        V: Evidence embeddings (B, n, d) or (n, d)
        q: Query embedding (B, d) or (d)
        eta: Perturbation radius
        alpha: Weight for risk combination
        eps: Small epsilon

    Returns:
        Dictionary with all metrics
    """
    return compute_risk_ga(V, q, eta, alpha, eps)
