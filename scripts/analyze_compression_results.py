#!/usr/bin/env python3
"""Analyze compression benchmark results"""

import json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

# Load detailed results
results = []
with open("results/compression_benchmark/detailed_results_20260107_184238.jsonl", "r") as f:
    for line in f:
        results.append(json.loads(line))

print(f"Loaded {len(results)} results")

# Extract metrics
labels = np.array([r["label"] for r in results])
C_scores = np.array([r["gard"]["C"] for r in results])
gisr_scores = np.array([r["gard"]["gisr"] for r in results])
gard_abstain = np.array([r["gard"]["abstain"] for r in results])

isr_scores = np.array([r["compression"]["isr"] for r in results])
delta_scores = np.array([r["compression"]["delta"] for r in results])
roh_scores = np.array([r["compression"]["roh"] for r in results])
lp_mad_scores = np.array([r["compression"]["lp_mad"] for r in results])
comp_abstain = np.array([r["compression"]["abstain"] for r in results])

# Compute metrics
metrics = {
    "gard": {
        "C_auroc": float(roc_auc_score(labels, C_scores)),
        "gisr_auroc": float(roc_auc_score(labels, -gisr_scores)),
        "abstain_recall": float(np.mean(gard_abstain[labels == 1])),
        "abstain_coverage": float(1 - np.mean(gard_abstain)),
    },
    "compression": {
        "ISR_auroc": float(roc_auc_score(labels, -isr_scores)),
        "RoH_auroc": float(roc_auc_score(labels, roh_scores)),
        "lp_mad_auroc": float(roc_auc_score(labels, lp_mad_scores)),
        "abstain_recall": float(np.mean(comp_abstain[labels == 1])),
        "abstain_coverage": float(1 - np.mean(comp_abstain)),
    },
    "correlation": {
        "C_vs_lp_mad": {
            "pearson": float(pearsonr(C_scores, lp_mad_scores)[0]),
            "spearman": float(spearmanr(C_scores, lp_mad_scores)[0]),
        },
        "gisr_vs_ISR": {
            "pearson": float(pearsonr(gisr_scores, isr_scores)[0]),
            "spearman": float(spearmanr(gisr_scores, isr_scores)[0]),
        },
        "margin_vs_delta": {
            "pearson": float(pearsonr([r["gard"]["margin"] for r in results], delta_scores)[0]),
            "spearman": float(spearmanr([r["gard"]["margin"] for r in results], delta_scores)[0]),
        },
    },
    "agreement": {
        "abstention_agreement": float(np.mean(gard_abstain == comp_abstain)),
        "both_abstain": float(np.mean(gard_abstain & comp_abstain)),
        "either_abstain": float(np.mean(gard_abstain | comp_abstain)),
        "gard_only": float(np.mean(gard_abstain & ~comp_abstain)),
        "compression_only": float(np.mean(~gard_abstain & comp_abstain)),
    },
}

# Save summary
summary = {
    "config": {
        "dataset": "data/clot_bench.jsonl",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "n_examples": len(results),
        "eta": 0.1,
        "C_hi": 0.42,
        "lam": 0.7,
        "t": 0.25,
        "m_permutations": 10,
        "max_new_tokens": 64,
        "logprob_batch_size": 4,
        "seed": 42,
    },
    "metrics": metrics,
}

with open("results/compression_benchmark/summary_20260107_184238.json", "w") as f:
    json.dump(summary, f, indent=2)

# Print results
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
print(f"\nSummary saved to: results/compression_benchmark/summary_20260107_184238.json")
print("="*80)
