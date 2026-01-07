#!/usr/bin/env python3
"""
Run synthetic validation experiments.

Validates GARD metrics on 4 synthetic scenarios:
1. Consensus (all aligned)
2. Conflict (contradictory)
3. Weakness (insufficient)
4. Mixed

Usage:
    python scripts/run_synth.py --d 768 --eta 0.1 --output results/synthetic/
"""

import argparse
import json
import torch
from pathlib import Path

from gard.experiments.synthetic import run_synthetic_validation
from gard.utils import set_seed, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run synthetic validation experiments"
    )

    parser.add_argument(
        "--d",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)",
    )

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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu, default: cuda)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/synthetic/",
        help="Output directory for results (default: results/synthetic/)",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Run validation
    print("\n" + "=" * 60)
    print("SYNTHETIC VALIDATION EXPERIMENT")
    print("=" * 60)
    print(f"Embedding dimension: {args.d}")
    print(f"Perturbation radius (eta): {args.eta}")
    print(f"Risk weight (alpha): {args.alpha}")
    print(f"Random seed: {args.seed}")
    print("=" * 60 + "\n")

    results = run_synthetic_validation(
        d=args.d,
        eta=args.eta,
        alpha=args.alpha,
        device=device,
    )

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "synthetic_validation.json"

    # Convert tensor values to Python types for JSON serialization
    results_serializable = {
        "n_scenarios": results["n_scenarios"],
        "n_matches": results["n_matches"],
        "accuracy": results["accuracy"],
        "config": {
            "d": args.d,
            "eta": args.eta,
            "alpha": args.alpha,
            "seed": args.seed,
        },
        "results": results["results"],
    }

    with open(output_file, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Validation Accuracy: {results['accuracy']:.1%}")
    print(f"Scenarios Passed: {results['n_matches']}/{results['n_scenarios']}")

    if results["accuracy"] >= 0.75:
        print("\n✓ Validation PASSED")
        return 0
    else:
        print("\n✗ Validation FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
