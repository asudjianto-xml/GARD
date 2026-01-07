"""
Utility functions for GARD benchmark.
"""

import torch
import numpy as np
import random
from typing import Union


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Union[str, torch.device] = "cuda") -> torch.device:
    """Get PyTorch device with fallback."""
    if isinstance(device, torch.device):
        return device

    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA not available, using CPU")
            return torch.device("cpu")

    return torch.device(device)


def generate_random_permutations(n: int, m: int, device: torch.device) -> torch.Tensor:
    """
    Generate m random permutations of {0, 1, ..., n-1}.

    Args:
        n: Size of permutation
        m: Number of permutations
        device: Device to create tensor on

    Returns:
        Permutation indices (m, n)
    """
    perms = torch.stack([
        torch.randperm(n, device=device) for _ in range(m)
    ])
    return perms


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.

    Args:
        a: Tensor (..., d)
        b: Tensor (..., d)
        eps: Small epsilon for numerical stability

    Returns:
        Cosine similarity (...)
    """
    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True).clamp(min=eps)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True).clamp(min=eps)

    a_normalized = a / a_norm
    b_normalized = b / b_norm

    return (a_normalized * b_normalized).sum(dim=-1)
