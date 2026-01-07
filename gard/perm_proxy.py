"""
Permutation Proxy Dispersion Baseline.

Exact implementation per Section 5.1 of instruction pack.

Key points:
1. Harmonic positional weighting: w_j = (1/j) / Σ(1/t)
2. For each permutation π: z(π) = Σ_j w_j · (v_π(j)^T q)
3. Compute p(π) = sigmoid(z(π))
4. Dispersion: p_std, p_mad
"""

import torch
from typing import Dict, Tuple
from .utils import generate_random_permutations


def compute_harmonic_weights(n: int, device: torch.device) -> torch.Tensor:
    """
    Compute harmonic positional weights.

    w_j = (1/j) / Σ_{t=1}^n (1/t)

    Args:
        n: Number of positions
        device: Device for tensor

    Returns:
        Weights tensor (n,) summing to 1.0
    """
    positions = torch.arange(1, n + 1, dtype=torch.float32, device=device)
    w = 1.0 / positions  # (n,)
    w = w / w.sum()  # Normalize
    return w


def permute_embeddings(V: torch.Tensor, perm_indices: torch.Tensor) -> torch.Tensor:
    """
    Apply permutations to embeddings.

    Args:
        V: Evidence embeddings (B, n, d)
        perm_indices: Permutation indices (m, n)

    Returns:
        Permuted embeddings (B, m, n, d)
    """
    B, n, d = V.shape
    m = perm_indices.shape[0]

    # Expand V: (B, n, d) -> (B, 1, n, d) -> (B, m, n, d)
    V_expanded = V.unsqueeze(1).expand(B, m, n, d)

    # Expand perm_indices: (m, n) -> (1, m, n, 1) -> (B, m, n, d)
    perm_expanded = perm_indices.unsqueeze(0).unsqueeze(-1).expand(B, m, n, d)

    # Gather: for each (b, i, j), take V[b, perm[i, j], :]
    V_permuted = torch.gather(V_expanded, 2, perm_expanded)

    return V_permuted


def compute_proxy_scores(
    V: torch.Tensor,
    q: torch.Tensor,
    perm_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute weighted proxy scores for permutations.

    z(π) = Σ_j w_j · (v_π(j)^T q)

    Args:
        V: Evidence embeddings (B, n, d)
        q: Query embedding (B, d)
        perm_indices: Permutation indices (m, n)
        weights: Harmonic weights (n,)

    Returns:
        Proxy scores (B, m)
    """
    # Permute embeddings: (B, m, n, d)
    V_permuted = permute_embeddings(V, perm_indices)

    # Compute alignments: s[b, i, j] = v_π(j)[b, i, j]^T q[b]
    # q: (B, d) -> (B, 1, 1, d)
    q_expanded = q.unsqueeze(1).unsqueeze(2)

    # Element-wise product and sum over d
    alignments = (V_permuted * q_expanded).sum(dim=-1)  # (B, m, n)

    # Apply harmonic weights: weights (n,) -> (1, 1, n)
    w_expanded = weights.unsqueeze(0).unsqueeze(0)

    # Weighted sum: z = Σ_j w_j * s_j
    z = (alignments * w_expanded).sum(dim=-1)  # (B, m)

    return z


def compute_proxy_probabilities(z: torch.Tensor) -> torch.Tensor:
    """
    Compute sigmoid probabilities from proxy scores.

    p(π) = sigmoid(z(π))

    Args:
        z: Proxy scores (B, m)

    Returns:
        Probabilities (B, m)
    """
    return torch.sigmoid(z)


def compute_dispersion_std(p: torch.Tensor) -> torch.Tensor:
    """
    Compute standard deviation dispersion.

    Args:
        p: Probabilities (B, m)

    Returns:
        Standard deviation (B,)
    """
    return p.std(dim=-1)


def compute_dispersion_mad(p: torch.Tensor) -> torch.Tensor:
    """
    Compute median absolute deviation dispersion.

    MAD = median(|p - median(p)|)

    Args:
        p: Probabilities (B, m)

    Returns:
        MAD (B,)
    """
    # Compute median along permutation dimension
    median_p = p.median(dim=-1, keepdim=True).values  # (B, 1)

    # Absolute deviations
    abs_dev = (p - median_p).abs()  # (B, m)

    # MAD
    mad = abs_dev.median(dim=-1).values  # (B,)

    return mad


def compute_perm_proxy_dispersion(
    V: torch.Tensor,
    q: torch.Tensor,
    m: int = 10,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute permutation proxy dispersion baseline.

    Full pipeline per Section 5.1:
    1. Generate m random permutations
    2. Compute harmonic positional weights
    3. For each permutation, compute weighted alignment score
    4. Apply sigmoid to get probabilities
    5. Compute dispersion (std, MAD)

    Args:
        V: Evidence embeddings (B, n, d) or (n, d)
        q: Query embedding (B, d) or (d)
        m: Number of permutations (default 10)
        device: Device for computation

    Returns:
        Dictionary with:
        - p_std: Standard deviation dispersion (B,) or scalar
        - p_mad: MAD dispersion (B,) or scalar
        - p: All probabilities (B, m) or (m,)
        - z: All proxy scores (B, m) or (m,)
    """
    # Handle single batch case
    if V.dim() == 2:
        V = V.unsqueeze(0)
        q = q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if device is None:
        device = V.device

    B, n, d = V.shape

    # Generate random permutations
    perm_indices = generate_random_permutations(n, m, device)  # (m, n)

    # Compute harmonic weights
    weights = compute_harmonic_weights(n, device)  # (n,)

    # Compute proxy scores
    z = compute_proxy_scores(V, q, perm_indices, weights)  # (B, m)

    # Compute probabilities
    p = compute_proxy_probabilities(z)  # (B, m)

    # Compute dispersion
    p_std = compute_dispersion_std(p)  # (B,)
    p_mad = compute_dispersion_mad(p)  # (B,)

    result = {
        "p_std": p_std,
        "p_mad": p_mad,
        "p": p,
        "z": z,
    }

    if squeeze:
        result = {k: v.squeeze(0) if v.dim() > 0 else v for k, v in result.items()}

    return result
