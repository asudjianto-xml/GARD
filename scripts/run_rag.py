#!/usr/bin/env python3
"""
Run RAG hallucination detection benchmark.

Compares:
- GARD (Geometric Algebra Risk Detection)
- Permutation Proxy (positional dispersion)
- Permutation Logprob (teacher-forced dispersion)
- Semantic Entropy (clustering-based)

Usage:
    # Run GARD + baselines on dataset
    python scripts/run_rag.py \
        --dataset data/rag_examples.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --methods gard perm_proxy \
        --output results/rag/

    # Run all methods with custom hyperparameters
    python scripts/run_rag.py \
        --dataset data/rag_examples.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --methods gard perm_proxy perm_logprob semantic_entropy \
        --eta 0.1 \
        --m-permutations 10 \
        --k-answers 10 \
        --max-examples 100 \
        --output results/rag/
"""

import argparse
import json
import pickle
import torch
from pathlib import Path
from datetime import datetime

from gard.qwen_backend import QwenBackend
from gard.dataset import load_dataset, filter_by_evidence_count, get_dataset_stats
from gard.experiments.rag import run_rag_experiment, aggregate_results
from gard.eval import compare_methods
from gard.plotting import create_full_report
from gard.utils import set_seed, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG hallucination detection benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset file",
    )

    parser.add_argument(
        "--min-evidence",
        type=int,
        default=2,
        help="Minimum number of evidence passages (default: 2)",
    )

    parser.add_argument(
        "--max-evidence",
        type=int,
        default=16,
        help="Maximum number of evidence passages (default: 16)",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (default: all)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-7B-Instruct)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu, default: cuda)",
    )

    # Method selection
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["gard", "perm_proxy"],
        choices=["gard", "perm_proxy", "perm_logprob", "semantic_entropy"],
        help="Methods to run (default: gard perm_proxy)",
    )

    # GARD hyperparameters
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Perturbation radius for GISR (default: 0.1)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for risk combination (default: 0.5)",
    )

    # Permutation hyperparameters
    parser.add_argument(
        "--m-permutations",
        type=int,
        default=10,
        help="Number of permutations for baselines (default: 10)",
    )

    # Semantic entropy hyperparameters
    parser.add_argument(
        "--k-answers",
        type=int,
        default=10,
        help="Number of answers for semantic entropy (default: 10)",
    )

    parser.add_argument(
        "--se-threshold",
        type=float,
        default=0.85,
        help="Clustering threshold for semantic entropy (default: 0.85)",
    )

    # Generation hyperparameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens for generation (default: 64)",
    )

    # Batch sizes
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=8,
        help="Batch size for embedding (default: 8)",
    )

    parser.add_argument(
        "--logprob-batch-size",
        type=int,
        default=4,
        help="Batch size for logprob computation (default: 4)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="results/rag/",
        help="Output directory for results (default: results/rag/)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    return parser.parse_args()


def print_header():
    """Print header."""
    print("\n" + "=" * 70)
    print(" " * 15 + "RAG HALLUCINATION DETECTION BENCHMARK")
    print("=" * 70)


def print_config(args, config):
    """Print configuration."""
    print("\nCONFIGURATION:")
    print("-" * 70)
    print(f"Dataset:          {args.dataset}")
    print(f"Model:            {args.model}")
    print(f"Methods:          {', '.join(args.methods)}")
    print(f"Device:           {args.device}")
    print(f"Max examples:     {args.max_examples if args.max_examples else 'All'}")
    print(f"Evidence range:   [{args.min_evidence}, {args.max_evidence}]")
    print("\nHYPERPARAMETERS:")
    print(f"  GARD eta:       {config['eta']}")
    print(f"  GARD alpha:     {config['alpha']}")
    print(f"  Permutations:   {config['m_permutations']}")
    print(f"  K answers:      {config['k_answers']}")
    print(f"  SE threshold:   {config['se_threshold']}")
    print(f"  Max tokens:     {config['max_new_tokens']}")
    print(f"  Seed:           {args.seed}")
    print("-" * 70 + "\n")


def main():
    """Main function."""
    args = parse_args()

    print_header()

    # Set random seed
    set_seed(args.seed)

    # Create config dictionary
    config = {
        "eta": args.eta,
        "alpha": args.alpha,
        "m_permutations": args.m_permutations,
        "k_answers": args.k_answers,
        "se_threshold": args.se_threshold,
        "max_new_tokens": args.max_new_tokens,
        "embed_batch_size": args.embed_batch_size,
        "logprob_batch_size": args.logprob_batch_size,
    }

    print_config(args, config)

    # Load dataset
    print("Loading dataset...")
    examples = load_dataset(args.dataset)

    # Filter by evidence count
    print(f"Filtering by evidence count [{args.min_evidence}, {args.max_evidence}]...")
    examples = filter_by_evidence_count(
        examples,
        min_n=args.min_evidence,
        max_n=args.max_evidence,
    )

    # Print dataset stats
    stats = get_dataset_stats(examples)
    print(f"\nDataset statistics:")
    print(f"  Total examples:     {stats['n_examples']}")
    print(f"  Hallucinations:     {stats['n_hallucinations']}")
    print(f"  Supported:          {stats['n_supported']}")
    print(f"  Hallucination rate: {stats['hallucination_rate']:.1%}")
    print(f"  Evidence count:     [{stats['n_evidence_min']}, {stats['n_evidence_max']}]")
    print(f"  Mean evidence:      {stats['n_evidence_mean']:.1f}\n")

    # Initialize model
    print(f"Loading model: {args.model}...")
    device = get_device(args.device)
    backend = QwenBackend(args.model, device=device)
    print("Model loaded.\n")

    # Run experiment
    results = run_rag_experiment(
        backend,
        examples,
        config,
        methods=args.methods,
        max_examples=args.max_examples,
    )

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(results)

    # Evaluate and compare
    print("Evaluating methods...")
    labels = aggregated.pop("labels")

    comparison_results = compare_methods(aggregated, labels)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for method_name, method_results in comparison_results.items():
        if method_name == "correlations":
            continue

        print(f"\n{method_name}:")
        print(f"  Accuracy:  {method_results['accuracy']:.3f}")
        print(f"  AUROC:     {method_results['auroc']:.3f}")
        print(f"  AUPRC:     {method_results['auprc']:.3f}")

        # Print analysis if available
        if "analysis" in method_results:
            analysis = method_results["analysis"]
            if "conflict_auroc" in analysis:
                print(f"  Conflict AUROC: {analysis['conflict_auroc']:.3f}")
            if "weakness_auroc" in analysis:
                print(f"  Weakness AUROC: {analysis['weakness_auroc']:.3f}")

    # Print correlations
    if "correlations" in comparison_results:
        print("\n" + "-" * 70)
        print("CORRELATIONS (Spearman ρ):")
        for pair_name, corr_data in comparison_results["correlations"].items():
            print(f"  {pair_name:30s}: ρ={corr_data['correlation']:6.3f}, p={corr_data['p_value']:.3e}")

    print("=" * 70 + "\n")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results (pickle)
    results_file = output_dir / f"results_{timestamp}.pkl"
    with open(results_file, "wb") as f:
        pickle.dump({
            "args": vars(args),
            "config": config,
            "results": results,
            "aggregated": aggregated,
            "comparison": comparison_results,
            "labels": labels,
        }, f)
    print(f"Raw results saved to: {results_file}")

    # Save summary (JSON)
    summary_file = output_dir / f"summary_{timestamp}.json"

    summary = {
        "config": config,
        "dataset_stats": stats,
        "methods": {},
    }

    for method_name, method_results in comparison_results.items():
        if method_name == "correlations":
            continue

        summary["methods"][method_name] = {
            "accuracy": float(method_results["accuracy"]),
            "auroc": float(method_results["auroc"]),
            "auprc": float(method_results["auprc"]),
        }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_dir = output_dir / f"plots_{timestamp}"

        # Prepare data for plotting - merge aggregated scores with comparison results
        plot_data = {}
        for method_name, method_data in aggregated.items():
            plot_data[method_name] = {
                "scores": method_data["scores"],
                "predictions": method_data["predictions"],
                "auroc": comparison_results[method_name]["auroc"],
                "auprc": comparison_results[method_name]["auprc"],
            }
            # Add optional fields
            if "coherence" in method_data:
                plot_data[method_name]["coherence"] = method_data["coherence"]
            if "gisr" in method_data:
                plot_data[method_name]["gisr"] = method_data["gisr"]
            if "abstention" in comparison_results[method_name]:
                plot_data[method_name]["abstention"] = comparison_results[method_name]["abstention"]
            if "analysis" in comparison_results[method_name]:
                plot_data[method_name]["analysis"] = comparison_results[method_name]["analysis"]

        # Add correlations back
        plot_comparison = plot_data.copy()
        plot_comparison["correlations"] = comparison_results.get("correlations", {})

        create_full_report(plot_comparison, labels, plot_dir)
        print(f"Plots saved to: {plot_dir}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
