#!/usr/bin/env python3
"""
Run GARD v2 evaluation on RAG datasets.

Implements two-gate decision rule (conflict vs weakness separation).
Compares GARD v2 against PermLogprob baseline (requires LLM generation).
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gard.qwen_backend import QwenBackend
from gard.gard_v2 import GARDv2, compute_coherence_v2, compute_margin_v2, compute_gisr_v2, compute_risk_score_v2
from gard.perm_logprob import compute_perm_logprob_dispersion


@dataclass
class ExampleResult:
    """Result for a single example."""
    example_id: str
    label: int
    subset_type: str

    # GARD v2
    C: float
    M: float
    absM: float
    gisr: float
    risk_score: float
    decision: str
    abstain: bool

    # PermLogprob_MAD
    perm_logprob_mad: float

    # Evidence count
    n_evidence: int


def load_dataset(path: str) -> List[Dict]:
    """Load JSONL dataset."""
    examples = []
    with open(path, 'r') as f:
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
    example: Dict,
    backend: QwenBackend,
    gard_v2: GARDv2,
    embed_batch_size: int = 8,
    m_permutations: int = 10,
    max_new_tokens: int = 64,
    logprob_batch_size: int = 4,
) -> ExampleResult:
    """Run GARD v2 and PermLogprob_MAD on a single example."""

    # Extract data
    example_id = example['id']
    query = example['query']

    # Handle both 'evidence' and 'contexts' fields
    if 'evidence' in example:
        if isinstance(example['evidence'], list) and len(example['evidence']) > 0:
            if isinstance(example['evidence'][0], dict):
                evidence = example['evidence']  # Keep as List[Dict]
                evidence_texts = [e['text'] for e in example['evidence']]
            else:
                # Convert plain text list to dict format
                evidence_texts = example['evidence']
                evidence = [{'text': text} for text in evidence_texts]
        else:
            evidence_texts = example['evidence']
            evidence = [{'text': text} for text in evidence_texts]
    else:
        evidence_texts = example.get('contexts', [])
        evidence = [{'text': text} for text in evidence_texts]

    label = example.get('label', 0)
    subset_type = example.get('subset_type', example.get('family', 'unknown'))

    n_evidence = len(evidence_texts)

    # Embed query and evidence
    all_texts = [query] + evidence_texts
    embeddings = embed_texts(backend, all_texts, embed_batch_size)

    q_emb = embeddings[0]  # (d,)
    V_emb = embeddings[1:]  # (n, d)

    # Run GARD v2
    gard_result = gard_v2.score_single(V_emb, q_emb)

    # Run PermLogprob_MAD (requires LLM generation for each permutation)
    perm_result = compute_perm_logprob_dispersion(
        backend,
        query,
        evidence,
        m=m_permutations,
        max_new_tokens=max_new_tokens,
        batch_size=logprob_batch_size,
    )
    perm_logprob_mad = perm_result['lp_mad'].item()

    return ExampleResult(
        example_id=example_id,
        label=label,
        subset_type=subset_type,
        C=gard_result['C'],
        M=gard_result['M'],
        absM=gard_result['absM'],
        gisr=gard_result['gisr'],
        risk_score=gard_result['risk_score'],
        decision=gard_result['decision'],
        abstain=gard_result['abstain'],
        perm_logprob_mad=perm_logprob_mad,
        n_evidence=n_evidence,
    )


def evaluate_results(results: List[ExampleResult]) -> Dict:
    """Compute evaluation metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    labels = np.array([r.label for r in results])

    # GARD v2 metrics
    gard_risks = np.array([r.risk_score for r in results])
    gard_abstains = np.array([int(r.abstain) for r in results])
    gard_decisions = np.array([r.decision for r in results])

    # PermLogprob_MAD metrics
    perm_mad = np.array([r.perm_logprob_mad for r in results])

    # Ranking metrics
    if len(np.unique(labels)) > 1:
        gard_auroc = roc_auc_score(labels, gard_risks)
        gard_auprc = average_precision_score(labels, gard_risks)

        perm_auroc = roc_auc_score(labels, perm_mad)
        perm_auprc = average_precision_score(labels, perm_mad)

        # Also compute -PermProxy for CLOT diagnostic
        perm_neg_auroc = roc_auc_score(labels, -perm_mad)
    else:
        gard_auroc = gard_auprc = float('nan')
        perm_auroc = perm_auprc = perm_neg_auroc = float('nan')

    # Decision metrics
    # Label 1 = unsafe (should abstain), Label 0 = safe (should accept)
    n_unsafe = (labels == 1).sum()
    n_safe = (labels == 0).sum()

    # Overall recall on unsafe
    unsafe_abstained = ((labels == 1) & (gard_abstains == 1)).sum()
    recall_unsafe = unsafe_abstained / n_unsafe if n_unsafe > 0 else float('nan')

    # Conflict recall
    conflict_mask = np.array([r.subset_type in ['balanced_contradiction', 'conflict'] for r in results])
    conflict_and_unsafe = (conflict_mask & (labels == 1))
    if conflict_and_unsafe.sum() > 0:
        conflict_abstained = ((gard_decisions == 'abstain_conflict') & conflict_and_unsafe).sum()
        conflict_recall = conflict_abstained / conflict_and_unsafe.sum()
    else:
        conflict_recall = float('nan')

    # Paraphrase acceptance
    paraphrase_mask = np.array([r.subset_type == 'paraphrase_explosion' for r in results])
    paraphrase_and_safe = (paraphrase_mask & (labels == 0))
    if paraphrase_and_safe.sum() > 0:
        paraphrase_accepted = ((gard_decisions == 'accept') & paraphrase_and_safe).sum()
        paraphrase_acceptance = paraphrase_accepted / paraphrase_and_safe.sum()
    else:
        paraphrase_acceptance = float('nan')

    # Coverage
    coverage = (gard_decisions == 'accept').sum() / len(results)

    # Accuracy (correct decision)
    # Correct if: (label==1 and abstain) or (label==0 and accept)
    correct = ((labels == 1) & (gard_abstains == 1)) | ((labels == 0) & (gard_abstains == 0))
    accuracy = correct.sum() / len(results)

    return {
        'gard_v2': {
            'auroc': gard_auroc,
            'auprc': gard_auprc,
            'accuracy': accuracy,
            'recall_unsafe': recall_unsafe,
            'conflict_recall': conflict_recall,
            'paraphrase_acceptance': paraphrase_acceptance,
            'coverage': coverage,
        },
        'perm_proxy_mad': {
            'auroc': perm_auroc,
            'auprc': perm_auprc,
            'auroc_neg': perm_neg_auroc,
        },
    }


def save_results(
    results: List[ExampleResult],
    metrics: Dict,
    config: Dict,
    output_dir: Path,
):
    """Save results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as CSV
    import pandas as pd

    df = pd.DataFrame([{
        'example_id': r.example_id,
        'label': r.label,
        'subset_type': r.subset_type,
        'C': r.C,
        'M': r.M,
        'absM': r.absM,
        'gisr': r.gisr,
        'risk_score': r.risk_score,
        'decision': r.decision,
        'abstain': r.abstain,
        'perm_logprob_mad': r.perm_logprob_mad,
        'n_evidence': r.n_evidence,
    } for r in results])

    csv_path = output_dir / 'detailed_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results to: {csv_path}")

    # Save metrics as JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'config': config,
            'metrics': metrics,
        }, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")

    # Save pickle for compatibility
    pkl_path = output_dir / 'results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'metrics': metrics,
            'config': config,
        }, f)
    print(f"✓ Saved pickle to: {pkl_path}")


def print_results(metrics: Dict, dataset_name: str):
    """Print results in readable format."""
    print("\n" + "="*80)
    print(f"GARD v2 RESULTS: {dataset_name}")
    print("="*80)

    gard = metrics['gard_v2']
    perm = metrics['perm_proxy_mad']

    print("\nGARD v2:")
    print(f"  AUROC:                 {gard['auroc']:.4f}")
    print(f"  AUPRC:                 {gard['auprc']:.4f}")
    print(f"  Accuracy:              {gard['accuracy']:.4f}")
    print(f"  Coverage:              {gard['coverage']:.4f}")
    print(f"  Unsafe Recall:         {gard['recall_unsafe']:.4f}")
    print(f"  Conflict Recall:       {gard['conflict_recall']:.4f}")
    print(f"  Paraphrase Acceptance: {gard['paraphrase_acceptance']:.4f}")

    print("\nPermLogprob_MAD:")
    print(f"  AUROC:                 {perm['auroc']:.4f}")
    print(f"  AUROC (negated):       {perm['auroc_neg']:.4f}")
    print(f"  AUPRC:                 {perm['auprc']:.4f}")

    print("\nGap (GARD v2 - PermLogprob_MAD):")
    print(f"  AUROC:                 {gard['auroc'] - perm['auroc']:+.4f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run GARD v2 evaluation")

    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset JSONL file')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Model name')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')

    # GARD v2 hyperparameters
    parser.add_argument('--eta', type=float, default=0.1,
                       help='Perturbation radius (default: 0.1)')
    parser.add_argument('--C-hi', type=float, default=0.35,
                       help='Conflict threshold (default: 0.35)')
    parser.add_argument('--lam', type=float, default=0.7,
                       help='Risk score lambda (default: 0.7)')
    parser.add_argument('--t', type=float, default=0.25,
                       help='Risk score temperature (default: 0.25)')

    # Experiment settings
    parser.add_argument('--max-examples', type=int, default=None,
                       help='Maximum examples to process')
    parser.add_argument('--embed-batch-size', type=int, default=8,
                       help='Batch size for embedding')
    parser.add_argument('--m-permutations', type=int, default=10,
                       help='Number of permutations for PermLogprob')
    parser.add_argument('--max-new-tokens', type=int, default=64,
                       help='Maximum tokens for generation (default: 64)')
    parser.add_argument('--logprob-batch-size', type=int, default=4,
                       help='Batch size for logprob computation (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("\n" + "="*80)
    print("GARD v2 EVALUATION")
    print("="*80)
    print(f"\nDataset:  {args.dataset}")
    print(f"Model:    {args.model}")
    print(f"Output:   {args.output}")
    print(f"\nGARD v2 Hyperparameters:")
    print(f"  eta:    {args.eta}")
    print(f"  C_hi:   {args.C_hi}")
    print(f"  lambda: {args.lam}")
    print(f"  t:      {args.t}")
    print(f"\nPermLogprob Baseline:")
    print(f"  m_permutations:      {args.m_permutations}")
    print(f"  max_new_tokens:      {args.max_new_tokens}")
    print(f"  logprob_batch_size:  {args.logprob_batch_size}")
    print(f"\nSeed:     {args.seed}")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    examples = load_dataset(args.dataset)
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Loaded {len(examples)} examples")

    # Load model
    print(f"\nLoading model: {args.model}...")
    backend = QwenBackend(args.model, device=args.device)
    print("Model loaded")

    # Initialize GARD v2
    gard_v2 = GARDv2(
        eta=args.eta,
        C_hi=args.C_hi,
        lam=args.lam,
        t=args.t,
        device=args.device,
    )

    # Run evaluation
    print(f"\nProcessing {len(examples)} examples...")
    results = []
    for example in tqdm(examples, desc="Processing"):
        result = run_example(
            example,
            backend,
            gard_v2,
            args.embed_batch_size,
            args.m_permutations,
            args.max_new_tokens,
            args.logprob_batch_size,
        )
        results.append(result)

    print(f"Completed {len(results)} examples")

    # Evaluate
    print("\nComputing metrics...")
    metrics = evaluate_results(results)

    # Save
    config = {
        'dataset': args.dataset,
        'model': args.model,
        'eta': args.eta,
        'C_hi': args.C_hi,
        'lam': args.lam,
        't': args.t,
        'm_permutations': args.m_permutations,
        'max_new_tokens': args.max_new_tokens,
        'logprob_batch_size': args.logprob_batch_size,
        'seed': args.seed,
    }

    output_dir = Path(args.output)
    save_results(results, metrics, config, output_dir)

    # Print
    dataset_name = Path(args.dataset).stem
    print_results(metrics, dataset_name)

    print("\n✓ Evaluation complete!\n")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
