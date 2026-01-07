"""
Permutation Logprob Dispersion Baseline.

Exact implementation per Section 5.1 of instruction pack.

Key points:
1. Generate canonical answer y* using identity permutation
2. For each permutation π, compute teacher-forced log P(y* | prompt_π)
3. Dispersion: lp_std, lp_mad
"""

import torch
from typing import Dict, List, Tuple
from .utils import generate_random_permutations
from .prompts import build_prompt_permuted


def generate_canonical_answer(
    backend,
    query: str,
    evidence: List[Dict[str, str]],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    """
    Generate canonical answer using identity permutation (temperature=0).

    Args:
        backend: QwenBackend instance
        query: Question string
        evidence: List of evidence dicts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 for greedy)

    Returns:
        Canonical answer string
    """
    from .prompts import build_prompt

    # Build prompt with identity permutation
    prompt = build_prompt(query, evidence)

    # Generate answer (greedy decoding)
    # Use do_sample=False for greedy decoding (temperature=0)
    answers = backend.generate(
        [prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=1.0,
        do_sample=(temperature > 0),
    )

    return answers[0].strip()


def compute_logprobs_for_permutations(
    backend,
    query: str,
    evidence: List[Dict[str, str]],
    canonical_answer: str,
    perm_indices: torch.Tensor,
    batch_size: int = 4,
) -> torch.Tensor:
    """
    Compute teacher-forced logprobs for all permutations.

    For each permutation π:
    1. Build prompt with permuted evidence
    2. Compute log P(y* | prompt_π)

    Args:
        backend: QwenBackend instance
        query: Question string
        evidence: List of evidence dicts
        canonical_answer: Canonical answer y*
        perm_indices: Permutation indices (m, n)
        batch_size: Batch size for processing

    Returns:
        Log probabilities (m,)
    """
    m = perm_indices.shape[0]
    n = perm_indices.shape[1]

    # Generate all permuted prompts
    prompts = []
    for i in range(m):
        perm = perm_indices[i].cpu().tolist()
        prompt = build_prompt_permuted(query, evidence, perm)
        prompts.append(prompt)

    # Compute logprobs in batches
    logprobs = []
    for i in range(0, m, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_answers = [canonical_answer] * len(batch_prompts)

        # Compute teacher-forced logprobs
        batch_logprobs = backend.logprob(batch_prompts, batch_answers)
        logprobs.append(batch_logprobs)

    # Concatenate all batches
    logprobs = torch.cat(logprobs, dim=0)  # (m,)

    return logprobs


def compute_dispersion_std_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute standard deviation for 1D tensor.

    Args:
        x: Values (m,)

    Returns:
        Standard deviation (scalar)
    """
    # Use unbiased=False to match numpy's default (ddof=0)
    # Handle edge case of single element
    if x.numel() <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return x.std(unbiased=False)


def compute_dispersion_mad_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute median absolute deviation for 1D tensor.

    MAD = median(|x - median(x)|)

    Args:
        x: Values (m,)

    Returns:
        MAD (scalar)
    """
    median_x = x.median()
    abs_dev = (x - median_x).abs()
    mad = abs_dev.median()
    return mad


def compute_perm_logprob_dispersion(
    backend,
    query: str,
    evidence: List[Dict[str, str]],
    m: int = 10,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    device: torch.device = None,
) -> Dict[str, any]:
    """
    Compute permutation logprob dispersion baseline.

    Full pipeline per Section 5.1:
    1. Generate canonical answer y* using identity permutation
    2. Sample m random permutations
    3. For each permutation, compute log P(y* | prompt_π)
    4. Compute dispersion (std, MAD)

    Args:
        backend: QwenBackend instance
        query: Question string
        evidence: List of evidence dicts
        m: Number of permutations (default 10)
        max_new_tokens: Maximum tokens for canonical answer
        batch_size: Batch size for logprob computation
        device: Device for computation

    Returns:
        Dictionary with:
        - lp_std: Standard deviation of logprobs
        - lp_mad: MAD of logprobs
        - lp: All logprobs (m,)
        - canonical_answer: Generated canonical answer y*
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(evidence)

    # Generate canonical answer
    canonical_answer = generate_canonical_answer(
        backend,
        query,
        evidence,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    # Generate random permutations
    perm_indices = generate_random_permutations(n, m, device)  # (m, n)

    # Compute logprobs for all permutations
    logprobs = compute_logprobs_for_permutations(
        backend,
        query,
        evidence,
        canonical_answer,
        perm_indices,
        batch_size=batch_size,
    )  # (m,)

    # Compute dispersion
    lp_std = compute_dispersion_std_1d(logprobs)
    lp_mad = compute_dispersion_mad_1d(logprobs)

    return {
        "lp_std": lp_std,
        "lp_mad": lp_mad,
        "lp": logprobs,
        "canonical_answer": canonical_answer,
    }


def compute_perm_logprob_dispersion_batch(
    backend,
    queries: List[str],
    evidence_lists: List[List[Dict[str, str]]],
    m: int = 10,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    device: torch.device = None,
) -> List[Dict[str, any]]:
    """
    Compute permutation logprob dispersion for batch of examples.

    Args:
        backend: QwenBackend instance
        queries: List of question strings
        evidence_lists: List of evidence lists
        m: Number of permutations per example
        max_new_tokens: Maximum tokens for canonical answers
        batch_size: Batch size for logprob computation
        device: Device for computation

    Returns:
        List of result dictionaries, one per example
    """
    results = []
    for query, evidence in zip(queries, evidence_lists):
        result = compute_perm_logprob_dispersion(
            backend,
            query,
            evidence,
            m=m,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
        )
        results.append(result)

    return results
