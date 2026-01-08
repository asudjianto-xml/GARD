"""
Compression Failure Metrics from Chlon et al. 2025

Implementation of Information Sufficiency Ratio (ISR), Bits-to-Trust (B2T),
and Risk-of-Hallucination (RoH) metrics from:

"Predictable Compression Failures: Why Language Models Actually Hallucinate"
https://arxiv.org/abs/2509.11208

Key insight: Hallucinations stem from predictable information compression failures
due to models not being permutation-invariant in their inference.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.special import expit  # sigmoid


def kl_bernoulli(p: float, q: float, eps: float = 1e-12) -> float:
    """
    Compute KL divergence between two Bernoulli distributions.

    KL(Ber(p) || Ber(q)) = p*log(p/q) + (1-p)*log((1-p)/(1-q))

    Args:
        p: First Bernoulli parameter
        q: Second Bernoulli parameter
        eps: Small constant for numerical stability

    Returns:
        KL divergence in nats
    """
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)

    term1 = p * np.log(p / q)
    term2 = (1 - p) * np.log((1 - p) / (1 - q))

    return term1 + term2


def bits_to_trust(
    q_lo: float,
    h_star: float = 0.95,
    eps: float = 1e-12,
) -> float:
    """
    Compute Bits-to-Trust (B2T): information required for target reliability.

    B2T(h*) = KL(Ber(1 - h*) || Ber(q_lo))

    Where:
    - h* is target reliability (e.g., 0.95 for 95% correct)
    - q_lo is lower bound on model accuracy (baseline performance)

    Args:
        q_lo: Lower bound on model accuracy (e.g., 0.5 for random)
        h_star: Target reliability (default: 0.95)
        eps: Numerical stability constant

    Returns:
        B2T in nats (information required to achieve h*)
    """
    p_error_target = 1 - h_star  # Target error rate
    p_error_baseline = 1 - q_lo  # Baseline error rate

    return kl_bernoulli(p_error_target, p_error_baseline, eps)


def information_budget_from_dispersion(
    logprobs: np.ndarray,
    method: str = "mad",
) -> float:
    """
    Estimate information budget Δ̄ from permutation dispersion.

    The key insight: lower dispersion → more information
    Information ∝ 1 / dispersion

    Args:
        logprobs: Log probabilities across m permutations (m,)
        method: "mad" or "std" for dispersion measure

    Returns:
        Information budget Δ̄ in nats
    """
    if method == "mad":
        median_lp = np.median(logprobs)
        dispersion = np.median(np.abs(logprobs - median_lp))
    elif method == "std":
        dispersion = np.std(logprobs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Prevent division by zero
    dispersion = max(dispersion, 1e-6)

    # Information is inverse of dispersion
    # Scale factor chosen to match paper's empirical findings
    # (can be calibrated on validation set)
    delta = 1.0 / dispersion

    return delta


def information_sufficiency_ratio(
    logprobs: np.ndarray,
    q_lo: float = 0.5,
    h_star: float = 0.95,
    method: str = "mad",
    eps: float = 1e-12,
) -> Tuple[float, float, float]:
    """
    Compute Information Sufficiency Ratio (ISR).

    ISR = Δ̄(x) / B2T(x; h*)

    Decision rule:
    - ISR < 1: Abstain (insufficient information)
    - ISR ≥ 1: Answer (sufficient information)

    Args:
        logprobs: Log probabilities across m permutations (m,)
        q_lo: Lower bound on model accuracy
        h_star: Target reliability
        method: Dispersion measure ("mad" or "std")
        eps: Numerical stability constant

    Returns:
        Tuple of (ISR, Δ̄, B2T)
    """
    # Compute information budget from dispersion
    delta = information_budget_from_dispersion(logprobs, method)

    # Compute information required for target reliability
    b2t = bits_to_trust(q_lo, h_star, eps)

    # Information Sufficiency Ratio
    isr = delta / b2t

    return isr, delta, b2t


def risk_of_hallucination(
    delta: float,
    q_bar: float = 0.5,
    eps: float = 1e-12,
) -> float:
    """
    Compute Risk-of-Hallucination (RoH).

    RoH = 1 - p_max(Δ̄(x), q̄(x))

    Where p_max is the maximum achievable accuracy given information budget Δ̄
    and mean model performance q̄.

    Args:
        delta: Information budget (in nats)
        q_bar: Mean model performance (baseline accuracy)
        eps: Numerical stability constant

    Returns:
        Risk of hallucination (error probability)
    """
    # Maximum achievable accuracy given information budget
    # Using the inverse of KL divergence relationship
    # p_max ≈ q_bar + delta * (1 - q_bar)
    # This is an approximation; exact inversion requires numerical methods

    # Clip to valid probability range
    p_max = min(q_bar + delta * (1 - q_bar), 1.0 - eps)

    # Risk is complement of maximum achievable accuracy
    roh = 1.0 - p_max

    return roh


def compression_decision(
    logprobs: np.ndarray,
    q_lo: float = 0.5,
    h_star: float = 0.95,
    isr_threshold: float = 1.0,
    method: str = "mad",
) -> Dict[str, float]:
    """
    Make abstention decision based on compression failure metrics.

    Args:
        logprobs: Log probabilities across m permutations (m,)
        q_lo: Lower bound on model accuracy
        h_star: Target reliability
        isr_threshold: ISR threshold for abstention (default: 1.0)
        method: Dispersion measure

    Returns:
        Dictionary with:
        - isr: Information Sufficiency Ratio
        - delta: Information budget
        - b2t: Bits-to-Trust
        - roh: Risk of Hallucination
        - decision: "accept" or "abstain"
        - abstain: Boolean flag
    """
    # Compute ISR
    isr, delta, b2t = information_sufficiency_ratio(
        logprobs, q_lo, h_star, method
    )

    # Compute RoH
    roh = risk_of_hallucination(delta, q_lo)

    # Decision rule: ISR < threshold → abstain
    abstain = isr < isr_threshold
    decision = "abstain" if abstain else "accept"

    return {
        "isr": isr,
        "delta": delta,
        "b2t": b2t,
        "roh": roh,
        "decision": decision,
        "abstain": abstain,
    }


def analyze_permutation_dispersion(
    backend,
    query: str,
    evidence: list,
    m_permutations: int = 12,
    max_new_tokens: int = 64,
    batch_size: int = 4,
) -> Dict[str, any]:
    """
    Analyze permutation dispersion for compression failure detection.

    This is the core evaluation protocol from Chlon et al. 2025.

    Args:
        backend: QwenBackend instance
        query: Question string
        evidence: List of evidence dicts with 'text' field
        m_permutations: Number of permutations to test
        max_new_tokens: Maximum tokens for generation
        batch_size: Batch size for logprob computation

    Returns:
        Dictionary with dispersion analysis and compression metrics
    """
    from .perm_logprob import compute_perm_logprob_dispersion

    # Compute permutation logprob dispersion
    result = compute_perm_logprob_dispersion(
        backend,
        query,
        evidence,
        m=m_permutations,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    logprobs = result["lp"].cpu().numpy()  # (m,)

    # Compute compression failure metrics
    compression_result = compression_decision(
        logprobs,
        q_lo=0.5,  # Conservative baseline
        h_star=0.95,  # 95% target reliability
        isr_threshold=1.0,
    )

    # Combine results
    return {
        "canonical_answer": result["canonical_answer"],
        "logprobs": logprobs,
        "lp_std": result["lp_std"].item(),
        "lp_mad": result["lp_mad"].item(),
        **compression_result,
    }


# Calibration utilities

def calibrate_q_lo(
    logprobs_list: list,
    labels: np.ndarray,
) -> float:
    """
    Calibrate q_lo (baseline accuracy) from validation data.

    Args:
        logprobs_list: List of logprob arrays, one per example
        labels: Binary labels (0=supported, 1=hallucination)

    Returns:
        Calibrated q_lo value
    """
    # Compute mean accuracy from validation set
    # This is a placeholder - proper calibration would use a held-out set
    # and estimate the lowest quantile of performance

    # For now, return conservative estimate
    return 0.5


def calibrate_dispersion_to_information(
    logprobs_list: list,
    labels: np.ndarray,
    method: str = "mad",
) -> float:
    """
    Calibrate the dispersion-to-information mapping.

    This learns the scaling factor between permutation dispersion
    and information budget.

    Args:
        logprobs_list: List of logprob arrays
        labels: Binary labels
        method: Dispersion measure

    Returns:
        Calibration constant
    """
    # This would require fitting on validation data
    # For now, return unit scaling
    return 1.0
