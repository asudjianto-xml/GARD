"""
Evaluation metrics for hallucination detection.

Exact implementation per Section 6 of instruction pack.

Metrics:
1. AUROC / AUPRC for binary classification
2. Abstention curves (fraction abstained vs accuracy)
3. Spearman correlation between methods
4. Conflict vs weakness analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_auroc_auprc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute AUROC and AUPRC.

    Args:
        scores: Uncertainty/risk scores (higher = more uncertain)
        labels: Binary labels (1 = hallucination, 0 = supported)

    Returns:
        Dictionary with auroc and auprc
    """
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    return {
        "auroc": auroc,
        "auprc": auprc,
    }


def compute_abstention_curve(
    scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_points: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute abstention curve.

    For various thresholds τ:
    - Abstain if score > τ
    - Compute accuracy on non-abstained examples
    - Compute fraction abstained

    Args:
        scores: Uncertainty/risk scores (higher = more uncertain)
        predictions: Binary predictions (1 = answer, 0 = abstain)
        labels: Ground truth labels (1 = hallucination, 0 = supported)
        n_points: Number of threshold points

    Returns:
        Dictionary with:
        - thresholds: Threshold values
        - frac_abstained: Fraction abstained at each threshold
        - accuracy: Accuracy on non-abstained at each threshold
        - auc: Area under abstention curve
    """
    # Sort scores in descending order
    sorted_indices = np.argsort(-scores)
    sorted_scores = scores[sorted_indices]
    sorted_preds = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]

    n = len(scores)

    # Generate thresholds
    thresholds = np.percentile(scores, np.linspace(0, 100, n_points))

    frac_abstained_list = []
    accuracy_list = []

    for tau in thresholds:
        # Abstain if score > tau
        abstain_mask = scores > tau
        n_abstained = abstain_mask.sum()
        frac_abstained = n_abstained / n

        # Accuracy on non-abstained
        if n_abstained < n:
            non_abstained_preds = predictions[~abstain_mask]
            non_abstained_labels = labels[~abstain_mask]

            # For hallucination detection:
            # Correct if: (pred=1 and label=0) or (pred=0 and label=1)
            # But typically predictions are already 0/1, so:
            # Accuracy = (predictions == labels).mean()
            accuracy = (non_abstained_preds == non_abstained_labels).mean()
        else:
            # All abstained
            accuracy = np.nan

        frac_abstained_list.append(frac_abstained)
        accuracy_list.append(accuracy)

    frac_abstained_arr = np.array(frac_abstained_list)
    accuracy_arr = np.array(accuracy_list)

    # Compute AUC (area under abstention curve)
    # Filter out NaN values
    valid_mask = ~np.isnan(accuracy_arr)
    if valid_mask.sum() > 1:
        auc = np.trapz(
            accuracy_arr[valid_mask],
            frac_abstained_arr[valid_mask],
        )
    else:
        auc = np.nan

    return {
        "thresholds": thresholds,
        "frac_abstained": frac_abstained_arr,
        "accuracy": accuracy_arr,
        "auc": auc,
    }


def compute_spearman_correlation(
    scores1: np.ndarray,
    scores2: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Spearman rank correlation between two score arrays.

    Args:
        scores1: First score array
        scores2: Second score array

    Returns:
        Dictionary with correlation and p-value
    """
    corr, pval = spearmanr(scores1, scores2)

    return {
        "correlation": corr,
        "p_value": pval,
    }


def compute_conflict_weakness_analysis(
    scores: np.ndarray,
    labels: np.ndarray,
    coherence_scores: Optional[np.ndarray] = None,
    gisr_scores: Optional[np.ndarray] = None,
) -> Dict[str, any]:
    """
    Analyze performance on conflict vs weakness cases.

    Conflict: High coherence (disagreement), low GISR (sufficient)
    Weakness: Low coherence (agreement), low GISR (insufficient)

    Args:
        scores: Uncertainty/risk scores
        labels: Binary labels (1 = hallucination)
        coherence_scores: Coherence values (optional)
        gisr_scores: GISR values (optional)

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Overall metrics
    overall_auroc = roc_auc_score(labels, scores)
    results["overall_auroc"] = overall_auroc

    # If coherence and GISR provided, analyze subgroups
    if coherence_scores is not None and gisr_scores is not None:
        # Define thresholds (median splits)
        coherence_median = np.median(coherence_scores)
        gisr_median = np.median(gisr_scores)

        # High coherence (conflict cases)
        high_coh_mask = coherence_scores > coherence_median
        if high_coh_mask.sum() > 10 and labels[high_coh_mask].sum() > 0:
            results["conflict_auroc"] = roc_auc_score(
                labels[high_coh_mask],
                scores[high_coh_mask],
            )
            results["conflict_n"] = high_coh_mask.sum()

        # Low GISR (insufficient evidence)
        low_gisr_mask = gisr_scores < gisr_median
        if low_gisr_mask.sum() > 10 and labels[low_gisr_mask].sum() > 0:
            results["weakness_auroc"] = roc_auc_score(
                labels[low_gisr_mask],
                scores[low_gisr_mask],
            )
            results["weakness_n"] = low_gisr_mask.sum()

        # Conflict cases: high coherence + low GISR
        conflict_mask = high_coh_mask & low_gisr_mask
        if conflict_mask.sum() > 10 and labels[conflict_mask].sum() > 0:
            results["pure_conflict_auroc"] = roc_auc_score(
                labels[conflict_mask],
                scores[conflict_mask],
            )
            results["pure_conflict_n"] = conflict_mask.sum()

    return results


def evaluate_method(
    scores: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    coherence_scores: Optional[np.ndarray] = None,
    gisr_scores: Optional[np.ndarray] = None,
    method_name: str = "Method",
) -> Dict[str, any]:
    """
    Comprehensive evaluation of a single method.

    Args:
        scores: Uncertainty/risk scores
        predictions: Binary predictions
        labels: Ground truth labels
        coherence_scores: Optional coherence values for analysis
        gisr_scores: Optional GISR values for analysis
        method_name: Name of method for reporting

    Returns:
        Dictionary with all evaluation metrics
    """
    results = {"method": method_name}

    # Basic accuracy
    accuracy = (predictions == labels).mean()
    results["accuracy"] = accuracy

    # AUROC/AUPRC
    roc_metrics = compute_auroc_auprc(scores, labels)
    results.update(roc_metrics)

    # Abstention curve
    abstention_metrics = compute_abstention_curve(scores, predictions, labels)
    results["abstention"] = abstention_metrics

    # Conflict/weakness analysis
    analysis = compute_conflict_weakness_analysis(
        scores, labels, coherence_scores, gisr_scores
    )
    results["analysis"] = analysis

    return results


def compare_methods(
    methods_dict: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
) -> Dict[str, any]:
    """
    Compare multiple methods.

    Args:
        methods_dict: Dictionary mapping method names to
                      {"scores": ..., "predictions": ..., ...}
        labels: Ground truth labels

    Returns:
        Dictionary with comparison results
    """
    results = {}

    # Evaluate each method
    for name, data in methods_dict.items():
        method_results = evaluate_method(
            scores=data["scores"],
            predictions=data["predictions"],
            labels=labels,
            coherence_scores=data.get("coherence"),
            gisr_scores=data.get("gisr"),
            method_name=name,
        )
        results[name] = method_results

    # Pairwise correlations
    method_names = list(methods_dict.keys())
    if len(method_names) >= 2:
        correlations = {}
        for i, name1 in enumerate(method_names):
            for name2 in method_names[i + 1 :]:
                scores1 = methods_dict[name1]["scores"]
                scores2 = methods_dict[name2]["scores"]

                corr_result = compute_spearman_correlation(scores1, scores2)
                correlations[f"{name1}_vs_{name2}"] = corr_result

        results["correlations"] = correlations

    return results
