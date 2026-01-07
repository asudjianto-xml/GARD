"""
RAG hallucination detection experiments.

Full pipeline integration:
1. Load dataset
2. Extract embeddings for all examples
3. Run all methods: GARD, perm_proxy, perm_logprob, semantic_entropy
4. Aggregate results
5. Evaluate and compare

Batch processing with progress tracking.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class RAGExperimentResult:
    """Results for single example."""
    example_id: str
    label: int  # 0 = supported, 1 = hallucination

    # GARD metrics
    gard_gisr: float
    gard_coherence: float
    gard_margin: float
    gard_risk: float
    gard_decision: bool  # True = accept

    # Permutation proxy
    perm_proxy_std: float
    perm_proxy_mad: float

    # Permutation logprob
    perm_logprob_std: Optional[float] = None
    perm_logprob_mad: Optional[float] = None

    # Semantic entropy
    semantic_entropy: Optional[float] = None
    n_clusters: Optional[int] = None


def run_gard_method(
    backend,
    example,
    eta: float = 0.1,
    alpha: float = 0.5,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Run GARD method on single example.

    Args:
        backend: QwenBackend instance
        example: Example from dataset
        eta: Perturbation radius
        alpha: Risk combination weight
        batch_size: Batch size for embedding

    Returns:
        Dictionary with GARD metrics
    """
    from ..embed import embed_texts
    from ..metrics import compute_all_gard_metrics
    from ..prompts import build_prompt

    # Extract evidence texts
    evidence_texts = [ev["text"] for ev in example.evidence]

    # Embed evidence
    V = embed_texts(backend, evidence_texts, batch_size=batch_size)  # (n, d)

    # Embed query
    q = embed_texts(backend, [example.query], batch_size=1)[0]  # (d,)

    # Compute GARD metrics
    metrics = compute_all_gard_metrics(V, q, eta=eta, alpha=alpha)

    return {
        "gisr": metrics["gisr"].item(),
        "coherence": metrics["C"].item(),
        "margin": metrics["M"].item(),
        "risk": metrics["risk_ga"].item(),
        "decision": metrics["decision_accept"].item(),
    }


def run_perm_proxy_method(
    backend,
    example,
    m: int = 10,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Run permutation proxy method on single example.

    Args:
        backend: QwenBackend instance
        example: Example from dataset
        m: Number of permutations
        batch_size: Batch size for embedding

    Returns:
        Dictionary with dispersion metrics
    """
    from ..embed import embed_texts
    from ..perm_proxy import compute_perm_proxy_dispersion

    # Extract evidence texts
    evidence_texts = [ev["text"] for ev in example.evidence]

    # Embed evidence
    V = embed_texts(backend, evidence_texts, batch_size=batch_size)  # (n, d)

    # Embed query
    q = embed_texts(backend, [example.query], batch_size=1)[0]  # (d,)

    # Compute permutation proxy dispersion
    result = compute_perm_proxy_dispersion(V, q, m=m)

    return {
        "p_std": result["p_std"].item(),
        "p_mad": result["p_mad"].item(),
    }


def run_perm_logprob_method(
    backend,
    example,
    m: int = 10,
    max_new_tokens: int = 64,
    batch_size: int = 4,
) -> Dict[str, float]:
    """
    Run permutation logprob method on single example.

    Args:
        backend: QwenBackend instance
        example: Example from dataset
        m: Number of permutations
        max_new_tokens: Maximum tokens for canonical answer
        batch_size: Batch size for logprob computation

    Returns:
        Dictionary with dispersion metrics
    """
    from ..perm_logprob import compute_perm_logprob_dispersion

    result = compute_perm_logprob_dispersion(
        backend,
        example.query,
        example.evidence,
        m=m,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    return {
        "lp_std": result["lp_std"].item(),
        "lp_mad": result["lp_mad"].item(),
    }


def run_semantic_entropy_method(
    backend,
    example,
    k: int = 10,
    threshold: float = 0.85,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> Dict[str, any]:
    """
    Run semantic entropy method on single example.

    Args:
        backend: QwenBackend instance
        example: Example from dataset
        k: Number of answers to generate
        threshold: Clustering similarity threshold
        max_new_tokens: Maximum tokens per answer
        batch_size: Batch size for embedding

    Returns:
        Dictionary with semantic entropy metrics
    """
    from ..semantic_entropy import compute_semantic_entropy

    result = compute_semantic_entropy(
        backend,
        example.query,
        example.evidence,
        k=k,
        threshold=threshold,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    return {
        "se": result["se"],
        "n_clusters": result["n_clusters"],
    }


def run_all_methods_on_example(
    backend,
    example,
    config: Dict[str, any],
    methods: List[str] = ["gard", "perm_proxy", "perm_logprob", "semantic_entropy"],
) -> RAGExperimentResult:
    """
    Run all methods on single example.

    Args:
        backend: QwenBackend instance
        example: Example from dataset
        config: Configuration dictionary with hyperparameters
        methods: List of methods to run

    Returns:
        RAGExperimentResult with all metrics
    """
    # Initialize result
    result = RAGExperimentResult(
        example_id=example.id,
        label=example.label,
        gard_gisr=0.0,
        gard_coherence=0.0,
        gard_margin=0.0,
        gard_risk=0.0,
        gard_decision=False,
        perm_proxy_std=0.0,
        perm_proxy_mad=0.0,
    )

    # Run GARD
    if "gard" in methods:
        gard_result = run_gard_method(
            backend,
            example,
            eta=config.get("eta", 0.1),
            alpha=config.get("alpha", 0.5),
            batch_size=config.get("embed_batch_size", 8),
        )
        result.gard_gisr = gard_result["gisr"]
        result.gard_coherence = gard_result["coherence"]
        result.gard_margin = gard_result["margin"]
        result.gard_risk = gard_result["risk"]
        result.gard_decision = gard_result["decision"]

    # Run perm_proxy
    if "perm_proxy" in methods:
        proxy_result = run_perm_proxy_method(
            backend,
            example,
            m=config.get("m_permutations", 10),
            batch_size=config.get("embed_batch_size", 8),
        )
        result.perm_proxy_std = proxy_result["p_std"]
        result.perm_proxy_mad = proxy_result["p_mad"]

    # Run perm_logprob
    if "perm_logprob" in methods:
        logprob_result = run_perm_logprob_method(
            backend,
            example,
            m=config.get("m_permutations", 10),
            max_new_tokens=config.get("max_new_tokens", 64),
            batch_size=config.get("logprob_batch_size", 4),
        )
        result.perm_logprob_std = logprob_result["lp_std"]
        result.perm_logprob_mad = logprob_result["lp_mad"]

    # Run semantic_entropy
    if "semantic_entropy" in methods:
        se_result = run_semantic_entropy_method(
            backend,
            example,
            k=config.get("k_answers", 10),
            threshold=config.get("se_threshold", 0.85),
            max_new_tokens=config.get("max_new_tokens", 64),
            batch_size=config.get("embed_batch_size", 8),
        )
        result.semantic_entropy = se_result["se"]
        result.n_clusters = se_result["n_clusters"]

    return result


def run_rag_experiment(
    backend,
    examples: List,
    config: Dict[str, any],
    methods: List[str] = ["gard", "perm_proxy"],
    max_examples: Optional[int] = None,
) -> List[RAGExperimentResult]:
    """
    Run full RAG experiment on dataset.

    Args:
        backend: QwenBackend instance
        examples: List of examples from dataset
        config: Configuration dictionary
        methods: List of methods to run
        max_examples: Optional limit on number of examples

    Returns:
        List of RAGExperimentResult
    """
    if max_examples is not None:
        examples = examples[:max_examples]

    print(f"Running RAG experiment on {len(examples)} examples...")
    print(f"Methods: {', '.join(methods)}")

    results = []
    for example in tqdm(examples, desc="Processing examples"):
        result = run_all_methods_on_example(
            backend,
            example,
            config,
            methods=methods,
        )
        results.append(result)

    print(f"Completed {len(results)} examples.")

    return results


def aggregate_results(
    results: List[RAGExperimentResult],
) -> Dict[str, np.ndarray]:
    """
    Aggregate results into arrays for evaluation.

    Args:
        results: List of experiment results

    Returns:
        Dictionary mapping method names to score/prediction arrays
    """
    n = len(results)

    # Extract labels
    labels = np.array([r.label for r in results])

    # GARD scores
    gard_gisr = np.array([r.gard_gisr for r in results])
    gard_risk = np.array([r.gard_risk for r in results])
    gard_coherence = np.array([r.gard_coherence for r in results])
    gard_decisions = np.array([1 if r.gard_decision else 0 for r in results])

    # Perm proxy scores
    perm_proxy_std = np.array([r.perm_proxy_std for r in results])
    perm_proxy_mad = np.array([r.perm_proxy_mad for r in results])

    aggregated = {
        "labels": labels,
        "GARD": {
            "scores": gard_risk,  # Higher risk = more uncertain
            "predictions": gard_decisions,  # 1 = accept, 0 = abstain
            "gisr": gard_gisr,
            "coherence": gard_coherence,
        },
        "PermProxy_STD": {
            "scores": perm_proxy_std,
            "predictions": (perm_proxy_std < np.median(perm_proxy_std)).astype(int),
        },
        "PermProxy_MAD": {
            "scores": perm_proxy_mad,
            "predictions": (perm_proxy_mad < np.median(perm_proxy_mad)).astype(int),
        },
    }

    # Add perm_logprob if available
    if results[0].perm_logprob_std is not None:
        perm_logprob_std = np.array([r.perm_logprob_std for r in results])
        perm_logprob_mad = np.array([r.perm_logprob_mad for r in results])

        aggregated["PermLogprob_STD"] = {
            "scores": perm_logprob_std,
            "predictions": (perm_logprob_std < np.median(perm_logprob_std)).astype(int),
        }
        aggregated["PermLogprob_MAD"] = {
            "scores": perm_logprob_mad,
            "predictions": (perm_logprob_mad < np.median(perm_logprob_mad)).astype(int),
        }

    # Add semantic entropy if available
    if results[0].semantic_entropy is not None:
        semantic_entropy = np.array([r.semantic_entropy for r in results])

        aggregated["SemanticEntropy"] = {
            "scores": semantic_entropy,
            "predictions": (semantic_entropy < np.median(semantic_entropy)).astype(int),
        }

    return aggregated
