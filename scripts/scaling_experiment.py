#!/usr/bin/env python3
"""
Scaling experiment: Compare GARD vs PermProxy_MAD at different dataset sizes.

Tests the hypothesis that GARD performs better on larger, more diverse datasets.
"""

import argparse
import json
import pickle
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate_dataset(n: int, output_path: str, seed: int = 42):
    """Generate synthetic dataset of size n."""
    cmd = [
        "python", "scripts/generate_synthetic_dataset.py",
        "--n", str(n),
        "--consensus", "0.5",
        "--conflict", "0.25",
        "--weakness", "0.15",
        "--output", output_path,
        "--seed", str(seed),
    ]

    print(f"\nGenerating dataset with {n} examples...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error generating dataset: {result.stderr}")
        return False

    print(result.stdout)
    return True


def run_benchmark(dataset_path: str, n: int, output_dir: str):
    """Run benchmark on dataset."""
    cmd = [
        "python", "scripts/run_rag.py",
        "--dataset", dataset_path,
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--methods", "gard", "perm_proxy",
        "--max-examples", str(n),
        "--output", output_dir,
        "--no-plots",  # Skip plots for speed
        "--seed", "42",
    ]

    print(f"\nRunning benchmark on {n} examples...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        return None

    # Extract summary file
    output_path = Path(output_dir)
    summary_files = list(output_path.glob("summary_*.json"))

    if not summary_files:
        print("No summary file found!")
        return None

    # Get most recent
    summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    return summary


def extract_results(summary: dict) -> dict:
    """Extract key metrics from summary."""
    return {
        'GARD': {
            'auroc': summary['methods']['GARD']['auroc'],
            'auprc': summary['methods']['GARD']['auprc'],
            'accuracy': summary['methods']['GARD']['accuracy'],
        },
        'PermProxy_STD': {
            'auroc': summary['methods']['PermProxy_STD']['auroc'],
            'auprc': summary['methods']['PermProxy_STD']['auprc'],
            'accuracy': summary['methods']['PermProxy_STD']['accuracy'],
        },
        'PermProxy_MAD': {
            'auroc': summary['methods']['PermProxy_MAD']['auroc'],
            'auprc': summary['methods']['PermProxy_MAD']['auprc'],
            'accuracy': summary['methods']['PermProxy_MAD']['accuracy'],
        },
        'hallucination_rate': summary['dataset_stats']['hallucination_rate'],
    }


def plot_scaling_results(results: dict, output_path: str):
    """Plot AUROC vs dataset size."""
    sizes = sorted(results.keys())

    gard_auroc = [results[n]['GARD']['auroc'] for n in sizes]
    proxy_std_auroc = [results[n]['PermProxy_STD']['auroc'] for n in sizes]
    proxy_mad_auroc = [results[n]['PermProxy_MAD']['auroc'] for n in sizes]

    gard_auprc = [results[n]['GARD']['auprc'] for n in sizes]
    proxy_std_auprc = [results[n]['PermProxy_STD']['auprc'] for n in sizes]
    proxy_mad_auprc = [results[n]['PermProxy_MAD']['auprc'] for n in sizes]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # AUROC plot
    ax1.plot(sizes, gard_auroc, 'o-', label='GARD', linewidth=2, markersize=8)
    ax1.plot(sizes, proxy_std_auroc, 's-', label='PermProxy_STD', linewidth=2, markersize=8)
    ax1.plot(sizes, proxy_mad_auroc, '^-', label='PermProxy_MAD', linewidth=2, markersize=8)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('Dataset Size', fontsize=12)
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title('AUROC vs Dataset Size', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.45, 1.0)

    # AUPRC plot
    ax2.plot(sizes, gard_auprc, 'o-', label='GARD', linewidth=2, markersize=8)
    ax2.plot(sizes, proxy_std_auprc, 's-', label='PermProxy_STD', linewidth=2, markersize=8)
    ax2.plot(sizes, proxy_mad_auprc, '^-', label='PermProxy_MAD', linewidth=2, markersize=8)
    ax2.set_xlabel('Dataset Size', fontsize=12)
    ax2.set_ylabel('AUPRC', fontsize=12)
    ax2.set_title('AUPRC vs Dataset Size', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    return fig


def print_summary_table(results: dict):
    """Print summary table of results."""
    print("\n" + "="*80)
    print("SCALING EXPERIMENT RESULTS")
    print("="*80)
    print(f"{'Size':<8} {'Method':<18} {'AUROC':>8} {'AUPRC':>8} {'Accuracy':>10}")
    print("-"*80)

    for size in sorted(results.keys()):
        res = results[size]

        print(f"{size:<8} {'GARD':<18} {res['GARD']['auroc']:>8.4f} {res['GARD']['auprc']:>8.4f} {res['GARD']['accuracy']:>10.4f}")
        print(f"{'':<8} {'PermProxy_STD':<18} {res['PermProxy_STD']['auroc']:>8.4f} {res['PermProxy_STD']['auprc']:>8.4f} {res['PermProxy_STD']['accuracy']:>10.4f}")
        print(f"{'':<8} {'PermProxy_MAD':<18} {res['PermProxy_MAD']['auroc']:>8.4f} {res['PermProxy_MAD']['auprc']:>8.4f} {res['PermProxy_MAD']['accuracy']:>10.4f}")
        print(f"{'':<8} {'Gap (GARD - MAD)':<18} {res['GARD']['auroc']-res['PermProxy_MAD']['auroc']:>8.4f} {res['GARD']['auprc']-res['PermProxy_MAD']['auprc']:>8.4f}")
        print("-"*80)

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Scaling experiment: GARD vs PermProxy_MAD")

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500],
        help="Dataset sizes to test (default: 50 100 200 500)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scaling/",
        help="Output directory for results (default: results/scaling/)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("SCALING EXPERIMENT: GARD vs PermProxy_MAD")
    print("="*80)
    print(f"Dataset sizes: {args.sizes}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("="*80)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run experiments for each size
    all_results = {}

    for n in args.sizes:
        # Generate dataset
        dataset_path = output_path / f"synthetic_{n}.jsonl"
        if not generate_dataset(n, str(dataset_path), args.seed):
            print(f"Skipping size {n} due to generation error")
            continue

        # Run benchmark
        benchmark_output = output_path / f"benchmark_{n}"
        summary = run_benchmark(str(dataset_path), n, str(benchmark_output))

        if summary is None:
            print(f"Skipping size {n} due to benchmark error")
            continue

        # Extract results
        results = extract_results(summary)
        all_results[n] = results

        print(f"\n✓ Completed size {n}:")
        print(f"  GARD AUROC: {results['GARD']['auroc']:.4f}")
        print(f"  PermProxy_MAD AUROC: {results['PermProxy_MAD']['auroc']:.4f}")
        print(f"  Gap: {results['GARD']['auroc'] - results['PermProxy_MAD']['auroc']:.4f}")

    # Save results
    results_file = output_path / "scaling_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    # Print summary
    print_summary_table(all_results)

    # Plot results
    plot_path = output_path / "scaling_plot.png"
    plot_scaling_results(all_results, str(plot_path))

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    sizes = sorted(all_results.keys())
    if len(sizes) >= 2:
        first_size = sizes[0]
        last_size = sizes[-1]

        gard_improvement = all_results[last_size]['GARD']['auroc'] - all_results[first_size]['GARD']['auroc']
        mad_improvement = all_results[last_size]['PermProxy_MAD']['auroc'] - all_results[first_size]['PermProxy_MAD']['auroc']

        print(f"From {first_size} to {last_size} examples:")
        print(f"  GARD AUROC improvement: {gard_improvement:+.4f}")
        print(f"  PermProxy_MAD AUROC improvement: {mad_improvement:+.4f}")
        print(f"  Relative improvement (GARD - MAD): {gard_improvement - mad_improvement:+.4f}")

        # Find crossover point
        gaps = [all_results[n]['GARD']['auroc'] - all_results[n]['PermProxy_MAD']['auroc'] for n in sizes]

        print(f"\nGaps (GARD - PermProxy_MAD):")
        for n, gap in zip(sizes, gaps):
            winner = "GARD" if gap > 0 else "PermProxy_MAD"
            print(f"  {n:4d} examples: {gap:+.4f} ({winner} wins)")

        # Check for crossover
        if any(g > 0 for g in gaps) and any(g < 0 for g in gaps):
            print(f"\n✓ Crossover detected! GARD advantage emerges with larger datasets.")
        elif all(g > 0 for g in gaps):
            print(f"\n✓ GARD consistently outperforms across all sizes.")
        else:
            print(f"\n✓ PermProxy_MAD consistently outperforms across all sizes.")

    print("="*80)
    print("\nExperiment complete!")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
