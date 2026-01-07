# GARD: Geometric Abstention via Robust Disagreement for Safe Large Language Models

[![Paper](https://img.shields.io/badge/Paper-SSRN-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6036774)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Geometric Algebra Framework for Hallucination Detection in Large Language Models**

## Abstract

Hallucination detection in large language models (LLMs) faces a fundamental tension: permutation-based uncertainty estimation methods optimize expected correctness under the model distribution but fail catastrophically on adversarial inputs, while existing geometric approaches conflate evidence disagreement with insufficient support. We introduce **GARD** (*Geometric Abstention via Robust Disagreement*), a framework grounded in Geometric Algebra that *separates* conflict detection from weakness detection via orthogonal geometric observables.

We prove that sign instability of invariant alignment margins defines a provably unsafe exclusion region, independent of application preferences. Evaluating on an adversarial benchmark designed to expose permutation method failures, we demonstrate that:
1. **Permutation dispersion is anti-correlated with adversarial risk** (below random performance) and provides no mechanism for safety-coverage tradeoff control
2. **GARD exposes a controllable Pareto frontier** where users explicitly trade conflict detection (ranging from near-complete at safety-first to moderate at coverage-first) against paraphrase acceptance
3. This tradeoff arises from empirical overlap in embedding space coherence, not theoretical insufficiency

As a risk-controlled abstention system rather than a ranking classifier, GARD provides deterministic linear-time computation with worst-case robustness guarantees and policy-adjustable safety constraints, offering a principled path toward auditable deployment in safety-critical applications.

---

## Key Concepts

### The Problem

Existing hallucination detection methods suffer from two critical flaws:

1. **Permutation-based methods** (PermProxy, semantic entropy) optimize **average-case correctness** $\mathbb{E}[\text{correct}|\text{model}]$, leading to catastrophic failures on adversarial inputs
2. **Standard geometric approaches** conflate two distinct failure modes:
   - **Evidence conflict**: Internal disagreement between chunks (e.g., "1879" vs. "1880")
   - **Insufficient support**: Weak alignment vulnerable to perturbations

### The Solution: Two-Gate Decision Rule

GARD separates these orthogonal risks via **Geometric Algebra observables**:

```
Decision(q) =
  if Coherence(C) ≥ C_hi:     abstain_conflict  (Conflict Gate)
  else if GISR < 1:           abstain_weak      (Weakness Gate)
  else:                       accept
```

- **Bivector Coherence (C)**: Measures geometric disagreement between evidence vectors
- **GISR (Geometric Insufficiency-Sufficiency Ratio)**: Measures margin robustness under bounded perturbations

### Key Innovation: Controllable Pareto Frontier

Unlike permutation methods that operate at a fixed point, GARD exposes a **safety-coverage tradeoff curve**:

| Threshold | Conflict Recall | Paraphrase Accept | Coverage | Use Case |
|-----------|----------------|-------------------|----------|----------|
| C_hi = 0.30 | **96.4%** | 0.0% | 2.5% | Medical, legal, finance |
| C_hi = 0.42 | 55.7% | **91.7%** | 58.5% | General QA |
| C_hi = 0.45 | 40.0% | **100.0%** | **72.0%** | Search engines |

**PermProxy-MAD**: 63.0% conflict recall, 80.0% paraphrase (AUROC 0.367, anti-correlated with risk)

---

## Installation

```bash
git clone https://github.com/asudjianto-xml/GARD.git
cd GARD

# Install package in development mode
pip install -e .

# Install required dependencies
pip install torch transformers accelerate bitsandbytes
pip install numpy scipy scikit-learn matplotlib tqdm
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU with ≥14GB VRAM (for Qwen2.5-7B)

---

## Quick Start

### Basic Usage

```python
from gard.qwen_backend import QwenBackend
from gard.embed import embed_texts
from gard.gard_v2 import GARDv2

# Initialize backend and GARD
backend = QwenBackend("Qwen/Qwen2.5-7B-Instruct", device="cuda")
gard = GARDv2(eta=0.1, C_hi=0.42, device="cuda")  # Balanced operating point

# Your data
query = "When was Einstein born?"
evidence_texts = [
    "Albert Einstein was born in 1879.",
    "Einstein's birth year is 1879.",
    "Some sources claim Einstein was born in 1880.",  # Conflict!
]

# Extract embeddings
V = embed_texts(backend, evidence_texts)  # (n, d)
q = embed_texts(backend, [query])[0]      # (d,)

# Compute GARD decision
result = gard.score_single(V, q)

print(f"Coherence (C): {result['C']:.3f}")
print(f"GISR: {result['gisr']:.3f}")
print(f"Risk Score: {result['risk_score']:.3f}")
print(f"Decision: {result['decision']}")  # 'abstain_conflict' or 'abstain_weak' or 'accept'
print(f"Should Abstain: {result['abstain']}")
```

---

## Replicating Paper Results

### Step 1: Generate CLOT Bench Dataset

The paper evaluates on **CLOT Bench** (Conflict + Lexical Overlap + Traps), a 200-example adversarial dataset with three families:

```bash
python scripts/generate_clot_bench.py \
    --output data/clot_bench.jsonl \
    --n-examples 200
```

Dataset breakdown:
- **Balanced Contradiction** (80 examples): Symmetric conflicts (e.g., "1879" vs. "1880")
- **Hybrid Trap+Refute** (60 examples): Authoritative traps + explicit refutations
- **Paraphrase Explosion** (60 examples): High surface diversity, semantic consistency

### Step 2: Run GARD v2 Evaluation

```bash
python scripts/run_gard_v2.py \
    --dataset data/clot_bench.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output results/gard_v2_clot_200/ \
    --sweep-thresholds  # Generate Pareto frontier
```

This produces:
- `detailed_results.csv` - Per-example metrics (C, GISR, decision)
- `pareto_frontier.csv` - Conflict recall vs. paraphrase acceptance for C_hi ∈ [0.30, 0.60]
- `distributions.png` - C and GISR distributions by family
- `tradeoff_curve.png` - Pareto frontier visualization
- `FINAL_SUMMARY.md` - Complete analysis

**Expected Results (Table 1 in paper):**

| Method | Conflict Recall | Paraphrase Accept | Coverage | AUROC |
|--------|----------------|-------------------|----------|-------|
| GARD (C_hi=0.30) | 96.4% | 0.0% | 2.5% | 0.583 |
| GARD (C_hi=0.42) | 55.7% | 91.7% | 58.5% | 0.583 |
| GARD (C_hi=0.45) | 40.0% | 100.0% | 72.0% | 0.583 |
| PermProxy-MAD | 63.0% | 80.0% | 58.0% | 0.367 |

### Step 3: Generate Paper Figures

**Figure 1: Distribution Analysis**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/gard_v2_clot_200/detailed_results.csv')

# Plot C and GISR distributions by family
# (Code provided in results/gard_v2_clot_200/FINAL_SUMMARY.md)
```

**Figure 2: Tradeoff Curve**
```python
df_pareto = pd.read_csv('results/gard_v2_clot_200/pareto_frontier.csv')

plt.figure(figsize=(10, 6))
plt.plot(df_pareto['para_acceptance'], df_pareto['conflict_recall'])
plt.scatter([0.0, 0.917, 1.0], [0.964, 0.557, 0.40],
            color='red', s=100, label='Operating Points')
plt.xlabel('Paraphrase Acceptance')
plt.ylabel('Conflict Recall')
plt.title('GARD Pareto Frontier: Safety-Coverage Tradeoff')
plt.legend()
plt.savefig('tradeoff_curve.png', dpi=300)
```

### Step 4: Compare with Baselines

```bash
python scripts/run_rag.py \
    --dataset data/clot_bench.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --methods gard perm_proxy semantic_entropy \
    --output results/comparison/
```

---

## Using GARD in Your Application

### Selecting an Operating Point

Choose `C_hi` based on your application's risk tolerance:

```python
# Safety-critical (medical, legal, finance)
gard = GARDv2(eta=0.1, C_hi=0.30, device="cuda")
# → 96.4% conflict recall, 2.5% coverage

# Balanced (general QA, research)
gard = GARDv2(eta=0.1, C_hi=0.42, device="cuda")
# → 55.7% conflict recall, 58.5% coverage

# High-throughput (search engines)
gard = GARDv2(eta=0.1, C_hi=0.45, device="cuda")
# → 40% conflict recall, 72% coverage
```

### Calibrating on Your Domain

```python
from gard.gard_v2 import calibrate_threshold

# Use labeled validation data
C_hi_optimal = calibrate_threshold(
    gard, validation_dataset,
    target_conflict_recall=0.95  # Your safety requirement
)

print(f"Optimal C_hi: {C_hi_optimal:.3f}")
```

### Integration with RAG Pipeline

```python
def safe_rag_pipeline(query, retrieved_docs, model, gard):
    """RAG pipeline with GARD safety check."""

    # Extract embeddings
    V = embed_texts(model, [doc['text'] for doc in retrieved_docs])
    q = embed_texts(model, [query])[0]

    # GARD decision
    result = gard.score_single(V, q)

    if result['abstain']:
        decision_type = result['decision']
        if decision_type == 'abstain_conflict':
            return {
                'answer': None,
                'reason': 'Evidence contains conflicting information',
                'coherence': result['C'],
                'suggestion': 'Please review source documents for contradictions'
            }
        elif decision_type == 'abstain_weak':
            return {
                'answer': None,
                'reason': 'Insufficient evidence support',
                'gisr': result['gisr'],
                'suggestion': 'Try providing more relevant documents'
            }

    # Safe to generate
    answer = model.generate(query, retrieved_docs)
    return {
        'answer': answer,
        'confidence': 1.0 - result['risk_score'],
        'metrics': result
    }
```

---

## Understanding the Metrics

### Bivector Coherence (C)

Measures **geometric disagreement** between evidence vectors via wedge products:

$$C = \frac{\|\\sum_{i<j} v_i \wedge v_j\|}{n(n-1)/2}$$

- **Low C** (< 0.3): Evidence is aligned, consistent
- **Medium C** (0.3-0.5): Some disagreement, requires inspection
- **High C** (> 0.5): Strong conflict, unsafe to answer

### GISR (Geometric Insufficiency-Sufficiency Ratio)

Measures **robustness to adversarial perturbations**:

$$\text{GISR} = \frac{|M|}{n\eta}$$

where $M = \sum_{i=1}^n v_i^\top q$ is the alignment margin.

- **GISR < 1**: Provably unsafe (perturbations can flip sign)
- **GISR ≥ 1**: Sufficient margin for robustness

### Risk Score

Combined metric for ranking:

$$\text{Risk} = \lambda \cdot C + (1-\lambda) \cdot \sigma\left(\frac{1-\text{GISR}}{t}\right)$$

Default: $\lambda = 0.7$, $t = 0.25$

---

## Repository Structure

```
GARD/
├── gard/                          # Core package
│   ├── gard_v2.py                # Two-gate GARD implementation
│   ├── perm_proxy.py             # PermProxy-MAD baseline
│   ├── semantic_entropy.py       # Semantic entropy baseline
│   ├── embed.py                  # Embedding extraction
│   ├── metrics.py                # Geometric metrics
│   ├── eval.py                   # Evaluation metrics
│   ├── plotting.py               # Visualizations
│   └── qwen_backend.py           # Model interface
├── scripts/
│   ├── generate_clot_bench.py   # Dataset generation
│   ├── run_gard_v2.py           # Main evaluation script
│   └── run_rag.py               # Baseline comparison
├── data/
│   ├── clot_bench.jsonl         # CLOT Bench (200 examples)
│   └── sample_rag.jsonl         # Sample data (20 examples)
├── results/
│   └── gard_v2_clot_200/        # Paper results
│       ├── detailed_results.csv
│       ├── pareto_frontier.csv
│       ├── distributions.png
│       ├── tradeoff_curve.png
│       └── FINAL_SUMMARY.md
├── paper/                        # Paper documentation
│   ├── REFERENCE_AUDIT_FINAL.md
│   ├── REVISION_SUMMARY.md
│   └── FRAMING_REVISION.md
└── tests/                        # Unit tests
```

---

## Computational Performance

### Complexity

- **GARD**: $O(n^2 d + nd)$ per example
- **PermProxy**: $O(mnd)$ per example (m = permutations)
- **Semantic Entropy**: $O(k \times T_{\text{gen}} + k^2)$ per example

### Timing (Qwen2.5-7B on A100)

| Method | Time per Example | 200 Examples |
|--------|-----------------|--------------|
| GARD | ~0.5s | ~2 minutes |
| PermProxy (m=10) | ~1s | ~3 minutes |
| Semantic Entropy (k=10) | ~15s | ~50 minutes |

**GARD is 30× faster than semantic entropy** while providing provable worst-case guarantees.

---

## Citation

If you use GARD in your research, please cite:

```bibtex
@article{sudjianto2025gard,
  title={GARD: Geometric Abstention via Robust Disagreement for Safe Large Language Models},
  author={Sudjianto, Agus and [Additional Authors]},
  journal={SSRN Electronic Journal},
  year={2025},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6036774}
}
```

**Paper:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6036774

---

## Key Results

### Main Finding: PermProxy Anti-Correlation

On CLOT Bench adversarial dataset:
- **PermProxy-MAD achieves 0.367 AUROC** (below random!)
- Confirms theoretical prediction: permutation dispersion optimizes $\mathbb{E}[\text{correct}|\text{model}]$, not worst-case safety
- Failure mechanism:
  - **Balanced Contradiction**: Model stable on wrong answer → low dispersion → false negative
  - **Paraphrase Explosion**: Surface variation → high dispersion → false positive

### GARD Controllability

**Safety-First Operating Point** (C_hi = 0.30):
- 96.4% conflict recall (catches nearly all adversarial conflicts)
- Appropriate for medical diagnosis, legal reasoning, financial advice
- Accepts very low coverage (2.5%) as acceptable cost for safety

**Balanced Operating Point** (C_hi = 0.42):
- 55.7% conflict recall, 91.7% paraphrase acceptance
- 58.5% overall coverage
- Suitable for general-purpose QA

**Coverage-First Operating Point** (C_hi = 0.45):
- 100% paraphrase acceptance, 72% coverage
- Misses 60% of conflicts
- Appropriate for search engines, exploratory applications

### Empirical Tradeoff Source

Distribution overlap in embedding space:
- **Balanced Contradiction**: C ∈ [0.223, 0.482], mean 0.376
- **Paraphrase Explosion**: C ∈ [0.395, 0.443], mean 0.411
- **Overlap region**: [0.395, 0.443] creates unavoidable tradeoff
- **GISR not discriminative**: All families have GISR ∈ [7.0, 8.8] (model is confident everywhere)

This is an **empirical constraint** of the embedding space, not a theoretical failure—GARD makes this tradeoff explicit and controllable.

---

## Troubleshooting

### Out of Memory Errors

Reduce batch sizes:
```bash
python scripts/run_gard_v2.py ... --embed-batch-size 4
```

### Model Loading Issues

Try smaller model:
```bash
python scripts/run_gard_v2.py --model Qwen/Qwen2-0.5B ...
```

### Slow Evaluation

Use GPU acceleration:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
python scripts/run_gard_v2.py ... --device cuda
```

---

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

Areas for contribution:
- Support for additional LLMs (Llama, Mistral, Gemma)
- Black-box API integration (OpenAI, Anthropic)
- Multi-GPU parallelization
- Additional baselines (conformal prediction, other uncertainty methods)
- Domain-specific calibration examples

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

We thank the reviewers for their helpful feedback. This work builds on:
- Geometric Algebra frameworks (Hestenes 1999, Brehmer et al. 2023)
- RAG methods (Lewis et al. 2020, Gao et al. 2023)
- Selective classification theory (Geifman & El-Yaniv 2017)
- Information-theoretic hallucination analysis (Chlon et al. 2025, Huang et al. 2023)

---

## Contact

For questions, issues, or collaboration:
- **GitHub Issues**: https://github.com/asudjianto-xml/GARD/issues
- **Paper**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6036774

---

**Status**: ✅ Complete implementation with verified paper results
**Last Updated**: January 2025
