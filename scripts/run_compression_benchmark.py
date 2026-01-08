#!/usr/bin/env python3
"""
Comparative Evaluation: GARD (Geometric) vs Compression Failures (Information-Theoretic)

Compares two approaches to hallucination detection:
1. GARD: Geometric Algebra (Bivector Coherence + GISR)
2. Chlon et al. 2025: Information Theory (ISR, B2T, RoH)

Both use permutation dispersion but interpret it differently:
- GARD: Geometric disagreement in embedding space
- Compression: Information sufficiency for reliable inference
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gard.qwen_backend import QwenBackend
from gard.gard_v2 import GARDv2
from gard.compression_metrics import analyze_permutation_dispersion


def load_clot_bench(data_path: str) -> List[Dict]:
    """Load CLOT Bench dataset."""
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def embed_texts(backend: QwenBackend, texts: List[str], batch_size: int = 8) -> torch.Tensor:
    """
    Embed texts using Qwen2.5 hidden states (mean pooling).

    Returns:
        Tensor of shape (len(texts), hidden_dim)
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Get hidden states
        hidden_states = backend.forward_hidden(batch)  # List of (T, d)

        # Mean pool over tokens
        batch_embs = []
        for h in hidden_states:
            # h is (T, d), take mean
            emb = h.mean(dim=0)  # (d,)
            batch_embs.append(emb)

        embeddings.extend(batch_embs)

    # Stack
    return torch.stack(embeddings)  # (n, d)


def run_example(
    backend: QwenBackend,
    gard_v2: GARDv2,
    example: Dict,
    m_permutations: int = 10,
    max_new_tokens: int = 64,
    logprob_batch_size: int = 4,
    embed_batch_size: int = 8,
) -> Dict:
    """
    Run both GARD and compression metrics on a single example.

    Returns:
        Dictionary with both geometric and information-theoretic metrics
    """
    # Extract data
    query = example.get("question", example.get("query"))

    # Handle evidence format (can be list of strings or list of dicts)
    if 'evidence' in example:
        if isinstance(example['evidence'], list) and len(example['evidence']) > 0:
            if isinstance(example['evidence'][0], dict):
                evidence = example['evidence']
                evidence_texts = [e['text'] for e in example['evidence']]
            else:
                evidence_texts = example['evidence']
                evidence = [{'text': text} for text in evidence_texts]
        else:
            evidence_texts = example['evidence']
            evidence = [{'text': text} for text in evidence_texts]
    else:
        evidence_texts = example.get('contexts', [])
        evidence = [{'text': text} for text in evidence_texts]

    label = example.get("label", 0)
    family = example.get("family", example.get("subset_type", "unknown"))

    # Embed query and evidence
    all_texts = [query] + evidence_texts
    embeddings = embed_texts(backend, all_texts, embed_batch_size)

    q_emb = embeddings[0]  # (d,)
    V_emb = embeddings[1:]  # (n, d)

    # Run GARD geometric abstention
    gard_result = gard_v2.score_single(V_emb, q_emb)

    # Run compression failure analysis (information-theoretic metrics)
    compression_result = analyze_permutation_dispersion(
        backend,
        query,
        evidence,
        m_permutations=m_permutations,
        max_new_tokens=max_new_tokens,
        batch_size=logprob_batch_size,
    )

    # Combine results (convert all numeric types to native Python for JSON serialization)
    return {
        "query": query,
        "label": int(label) if isinstance(label, (np.integer, np.bool_)) else label,
        "family": family,

        # GARD geometric metrics
        "gard": {
            "C": float(gard_result["C"]),
            "gisr": float(gard_result["gisr"]),
            "margin": float(gard_result["M"]),
            "risk_score": float(gard_result["risk_score"]),
            "decision": gard_result["decision"],
            "abstain": bool(gard_result["abstain"]),
        },

        # Compression information-theoretic metrics
        "compression": {
            "isr": float(compression_result["isr"]),
            "delta": float(compression_result["delta"]),
            "b2t": float(compression_result["b2t"]),
            "roh": float(compression_result["roh"]),
            "lp_mad": float(compression_result["lp_mad"]),
            "lp_std": float(compression_result["lp_std"]),
            "decision": compression_result["decision"],
            "abstain": bool(compression_result["abstain"]),
        },

        "canonical_answer": compression_result["canonical_answer"],
    }


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute evaluation metrics for both approaches.

    Metrics:
    - AUROC for each decision signal
    - Correlation between geometric and information-theoretic metrics
    - Agreement rate between abstention decisions
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr, spearmanr

    labels = np.array([r["label"] for r in results])

    # GARD signals
    C_scores = np.array([r["gard"]["C"] for r in results])
    gisr_scores = np.array([r["gard"]["gisr"] for r in results])
    gard_abstain = np.array([r["gard"]["abstain"] for r in results])

    # Compression signals
    isr_scores = np.array([r["compression"]["isr"] for r in results])
    delta_scores = np.array([r["compression"]["delta"] for r in results])
    roh_scores = np.array([r["compression"]["roh"] for r in results])
    lp_mad_scores = np.array([r["compression"]["lp_mad"] for r in results])
    comp_abstain = np.array([r["compression"]["abstain"] for r in results])

    # Compute AUROC for each signal
    # Higher risk → want higher signal for AUROC computation
    metrics = {
        "gard": {
            "C_auroc": roc_auc_score(labels, C_scores),  # Higher C → conflict
            "gisr_auroc": roc_auc_score(labels, -gisr_scores),  # Lower GISR → risk
            "abstain_recall": np.mean(gard_abstain[labels == 1]),
            "abstain_coverage": 1 - np.mean(gard_abstain),
        },
        "compression": {
            "ISR_auroc": roc_auc_score(labels, -isr_scores),  # Lower ISR → risk
            "RoH_auroc": roc_auc_score(labels, roh_scores),  # Higher RoH → risk
            "lp_mad_auroc": roc_auc_score(labels, lp_mad_scores),  # Higher MAD → risk
            "abstain_recall": np.mean(comp_abstain[labels == 1]),
            "abstain_coverage": 1 - np.mean(comp_abstain),
        },
        "correlation": {
            # Correlation between key metrics
            "C_vs_lp_mad": {
                "pearson": pearsonr(C_scores, lp_mad_scores)[0],
                "spearman": spearmanr(C_scores, lp_mad_scores)[0],
            },
            "gisr_vs_ISR": {
                "pearson": pearsonr(gisr_scores, isr_scores)[0],
                "spearman": spearmanr(gisr_scores, isr_scores)[0],
            },
            "margin_vs_delta": {
                "pearson": pearsonr(
                    [r["gard"]["margin"] for r in results],
                    delta_scores
                )[0],
                "spearman": spearmanr(
                    [r["gard"]["margin"] for r in results],
                    delta_scores
                )[0],
            },
        },
        "agreement": {
            "abstention_agreement": np.mean(gard_abstain == comp_abstain),
            "both_abstain": np.mean(gard_abstain & comp_abstain),
            "either_abstain": np.mean(gard_abstain | comp_abstain),
            "gard_only": np.mean(gard_abstain & ~comp_abstain),
            "compression_only": np.mean(~gard_abstain & comp_abstain),
        },
    }

    return metrics


def compute_family_breakdown(results: List[Dict]) -> Dict:
    """Compute metrics broken down by CLOT Bench family."""
    families = set(r["family"] for r in results)
    breakdown = {}

    for family in families:
        family_results = [r for r in results if r["family"] == family]
        if len(family_results) > 0:
            breakdown[family] = compute_metrics(family_results)

    return breakdown


def main():
    parser = argparse.ArgumentParser(
        description="Compare GARD geometric vs compression information-theoretic metrics"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/clot_bench.jsonl",
        help="Path to CLOT Bench dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/compression_benchmark",
        help="Output directory"
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)"
    )

    # GARD parameters
    parser.add_argument("--eta", type=float, default=0.1, help="Perturbation bound")
    parser.add_argument("--C-hi", type=float, default=0.42, help="Coherence threshold")
    parser.add_argument("--lam", type=float, default=0.7, help="Margin weight")
    parser.add_argument("--t", type=float, default=0.25, help="Margin threshold")

    # Permutation parameters
    parser.add_argument("--m-permutations", type=int, default=10, help="Number of permutations")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens for generation")
    parser.add_argument("--logprob-batch-size", type=int, default=4, help="Batch size for logprob computation")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data}")
    examples = load_clot_bench(args.data)
    if args.n_examples:
        examples = examples[:args.n_examples]
    print(f"Loaded {len(examples)} examples")

    # Initialize backend
    print(f"Loading model: {args.model}")
    backend = QwenBackend(args.model)

    # Initialize GARD v2
    print("Initializing GARD v2...")
    gard_v2 = GARDv2(
        eta=args.eta,
        C_hi=args.C_hi,
        lam=args.lam,
        t=args.t,
    )

    # Run evaluation
    print("Running comparative evaluation...")
    results = []

    for example in tqdm(examples, desc="Evaluating"):
        try:
            result = run_example(
                backend,
                gard_v2,
                example,
                m_permutations=args.m_permutations,
                max_new_tokens=args.max_new_tokens,
                logprob_batch_size=args.logprob_batch_size,
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    print(f"Successfully evaluated {len(results)} examples")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results)

    # Compute family breakdown
    print("Computing family breakdown...")
    family_breakdown = compute_family_breakdown(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = output_dir / f"detailed_results_{timestamp}.jsonl"
    print(f"Saving detailed results to {results_file}")
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Save summary metrics
    summary = {
        "config": {
            "dataset": args.data,
            "model": args.model,
            "n_examples": len(results),
            "eta": args.eta,
            "C_hi": args.C_hi,
            "lam": args.lam,
            "t": args.t,
            "m_permutations": args.m_permutations,
            "max_new_tokens": args.max_new_tokens,
            "logprob_batch_size": args.logprob_batch_size,
            "seed": args.seed,
        },
        "metrics": metrics,
        "family_breakdown": family_breakdown,
    }

    summary_file = output_dir / f"summary_{timestamp}.json"
    print(f"Saving summary to {summary_file}")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print key results
    print("\n" + "="*80)
    print("COMPARATIVE EVALUATION RESULTS")
    print("="*80)

    print("\n1. AUROC Scores (Risk Detection)")
    print("-" * 40)
    print(f"GARD Coherence (C):           {metrics['gard']['C_auroc']:.4f}")
    print(f"GARD GISR (margin robustness): {metrics['gard']['gisr_auroc']:.4f}")
    print(f"Compression ISR:               {metrics['compression']['ISR_auroc']:.4f}")
    print(f"Compression RoH:               {metrics['compression']['RoH_auroc']:.4f}")
    print(f"Compression LP-MAD:            {metrics['compression']['lp_mad_auroc']:.4f}")

    print("\n2. Abstention Performance")
    print("-" * 40)
    print(f"GARD - Unsafe Recall:          {metrics['gard']['abstain_recall']:.4f}")
    print(f"GARD - Coverage:               {metrics['gard']['abstain_coverage']:.4f}")
    print(f"Compression - Unsafe Recall:   {metrics['compression']['abstain_recall']:.4f}")
    print(f"Compression - Coverage:        {metrics['compression']['abstain_coverage']:.4f}")

    print("\n3. Correlation Between Approaches")
    print("-" * 40)
    print(f"C (geometric) vs LP-MAD (dispersion):")
    print(f"  Pearson:  {metrics['correlation']['C_vs_lp_mad']['pearson']:.4f}")
    print(f"  Spearman: {metrics['correlation']['C_vs_lp_mad']['spearman']:.4f}")
    print(f"GISR (geometric sufficiency) vs ISR (information sufficiency):")
    print(f"  Pearson:  {metrics['correlation']['gisr_vs_ISR']['pearson']:.4f}")
    print(f"  Spearman: {metrics['correlation']['gisr_vs_ISR']['spearman']:.4f}")

    print("\n4. Abstention Decision Agreement")
    print("-" * 40)
    print(f"Agreement Rate:                {metrics['agreement']['abstention_agreement']:.4f}")
    print(f"Both Abstain:                  {metrics['agreement']['both_abstain']:.4f}")
    print(f"Either Abstains:               {metrics['agreement']['either_abstain']:.4f}")
    print(f"GARD Only:                     {metrics['agreement']['gard_only']:.4f}")
    print(f"Compression Only:              {metrics['agreement']['compression_only']:.4f}")

    print("\n" + "="*80)
    print(f"\nResults saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
