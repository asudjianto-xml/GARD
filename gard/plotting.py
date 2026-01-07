"""
Plotting and visualization for evaluation results.

Exact implementation per Section 6 of instruction pack.

Plots:
1. ROC curves (AUROC)
2. Precision-Recall curves (AUPRC)
3. Abstention curves
4. Correlation scatter plots
5. Conflict vs weakness visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_roc_curves(
    methods_dict: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot ROC curves for multiple methods.

    Args:
        methods_dict: Dictionary mapping method names to
                      {"scores": ..., "auroc": ...}
        labels: Ground truth labels
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, data in methods_dict.items():
        scores = data["scores"]
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = data.get("auroc", 0.0)

        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})", linewidth=2)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Hallucination Detection", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_pr_curves(
    methods_dict: Dict[str, Dict[str, np.ndarray]],
    labels: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple methods.

    Args:
        methods_dict: Dictionary mapping method names to
                      {"scores": ..., "auprc": ...}
        labels: Ground truth labels
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    baseline = labels.mean()  # Fraction of positive class

    for name, data in methods_dict.items():
        scores = data["scores"]
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = data.get("auprc", 0.0)

        ax.plot(recall, precision, label=f"{name} (AUPRC={auprc:.3f})", linewidth=2)

    # Baseline reference line
    ax.axhline(baseline, color="k", linestyle="--", linewidth=1, label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves - Hallucination Detection", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_abstention_curves(
    methods_dict: Dict[str, Dict[str, any]],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot abstention curves for multiple methods.

    Args:
        methods_dict: Dictionary mapping method names to
                      {"abstention": {"frac_abstained": ..., "accuracy": ...}}
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, data in methods_dict.items():
        abstention_data = data.get("abstention", {})
        frac_abstained = abstention_data.get("frac_abstained")
        accuracy = abstention_data.get("accuracy")

        if frac_abstained is not None and accuracy is not None:
            # Filter out NaN values
            valid_mask = ~np.isnan(accuracy)
            ax.plot(
                frac_abstained[valid_mask],
                accuracy[valid_mask],
                label=name,
                linewidth=2,
                marker="o",
                markersize=4,
            )

    ax.set_xlabel("Fraction Abstained", fontsize=12)
    ax.set_ylabel("Accuracy on Non-Abstained", fontsize=12)
    ax.set_title("Abstention Curves", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_correlation_scatter(
    scores1: np.ndarray,
    scores2: np.ndarray,
    name1: str,
    name2: str,
    correlation: float,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Plot scatter plot of scores from two methods.

    Args:
        scores1: Scores from method 1
        scores2: Scores from method 2
        name1: Name of method 1
        name2: Name of method 2
        correlation: Spearman correlation value
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(scores1, scores2, alpha=0.5, s=30)

    ax.set_xlabel(f"{name1} Score", fontsize=12)
    ax.set_ylabel(f"{name2} Score", fontsize=12)
    ax.set_title(f"Score Correlation (ρ={correlation:.3f})", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add diagonal reference line
    min_val = min(scores1.min(), scores2.min())
    max_val = max(scores1.max(), scores2.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_conflict_weakness_breakdown(
    methods_dict: Dict[str, Dict[str, any]],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot AUROC breakdown for conflict vs weakness cases.

    Args:
        methods_dict: Dictionary mapping method names to
                      {"analysis": {"conflict_auroc": ..., "weakness_auroc": ...}}
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = []
    overall_scores = []
    conflict_scores = []
    weakness_scores = []

    for name, data in methods_dict.items():
        analysis = data.get("analysis", {})

        overall = analysis.get("overall_auroc")
        conflict = analysis.get("conflict_auroc")
        weakness = analysis.get("weakness_auroc")

        if overall is not None:
            methods.append(name)
            overall_scores.append(overall)
            conflict_scores.append(conflict if conflict is not None else 0)
            weakness_scores.append(weakness if weakness is not None else 0)

    x = np.arange(len(methods))
    width = 0.25

    ax.bar(x - width, overall_scores, width, label="Overall", alpha=0.8)
    ax.bar(x, conflict_scores, width, label="Conflict (High Coh)", alpha=0.8)
    ax.bar(x + width, weakness_scores, width, label="Weakness (Low GISR)", alpha=0.8)

    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Performance Breakdown: Conflict vs Weakness", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_full_report(
    comparison_results: Dict[str, any],
    labels: np.ndarray,
    output_dir: str,
) -> None:
    """
    Create full visualization report.

    Args:
        comparison_results: Results from compare_methods()
        labels: Ground truth labels
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    methods_dict = {}
    for name, results in comparison_results.items():
        if name != "correlations":
            methods_dict[name] = results

    # 1. ROC curves
    print("Generating ROC curves...")
    plot_roc_curves(
        methods_dict,
        labels,
        save_path=output_path / "roc_curves.png",
    )
    plt.close()

    # 2. PR curves
    print("Generating PR curves...")
    plot_pr_curves(
        methods_dict,
        labels,
        save_path=output_path / "pr_curves.png",
    )
    plt.close()

    # 3. Abstention curves
    print("Generating abstention curves...")
    plot_abstention_curves(
        methods_dict,
        save_path=output_path / "abstention_curves.png",
    )
    plt.close()

    # 4. Conflict/weakness breakdown
    print("Generating conflict/weakness breakdown...")
    plot_conflict_weakness_breakdown(
        methods_dict,
        save_path=output_path / "conflict_weakness.png",
    )
    plt.close()

    # 5. Correlation scatter plots
    if "correlations" in comparison_results:
        print("Generating correlation scatter plots...")
        correlations = comparison_results["correlations"]

        for pair_name, corr_data in correlations.items():
            name1, name2 = pair_name.replace("_vs_", " ").split()

            scores1 = methods_dict[name1]["scores"]
            scores2 = methods_dict[name2]["scores"]
            correlation = corr_data["correlation"]

            plot_correlation_scatter(
                scores1,
                scores2,
                name1,
                name2,
                correlation,
                save_path=output_path / f"correlation_{name1}_{name2}.png",
            )
            plt.close()

    print(f"All plots saved to {output_dir}")
