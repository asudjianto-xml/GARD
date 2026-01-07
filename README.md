# GARD Head-to-Head Benchmark: Qwen2.5-7B

**Status**: ✅ Full Implementation Complete

## Overview

Comprehensive benchmark comparing hallucination detection methods for RAG systems:

1. **GARD** (Geometric Algebra Risk Detection) - GISR + Coherence
2. **Permutation Baselines** - Proxy + Logprob Dispersion
3. **Semantic Entropy** - Generation + Clustering

Using **Qwen2.5-7B-Instruct** with internal hidden states on GPU.

## Features

✅ **Complete Implementation** (~3,200 lines)
- All 15 core modules implemented per specification
- Full test suite (570+ lines, 40+ tests)
- Two experiment runners with CLI
- Comprehensive evaluation metrics
- Publication-ready visualizations

## Installation

```bash
cd /home/asudjianto/jupyterlab/gard_head2head_qwen

# Install package
pip install -e .

# Required dependencies
pip install torch transformers accelerate bitsandbytes
pip install numpy scipy scikit-learn matplotlib tqdm
pip install pytest
```

## Quick Start

### 1. Synthetic Validation

Validate GARD metrics on 4 theoretical scenarios:

```bash
python scripts/run_synth.py \
    --d 768 \
    --eta 0.1 \
    --output results/synthetic/
```

Expected output: 75-100% validation accuracy

### 2. RAG Benchmark (Quick Test)

Run on sample dataset with GARD + permutation proxy:

```bash
python scripts/run_rag.py \
    --dataset data/sample_rag.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --methods gard perm_proxy \
    --max-examples 20 \
    --output results/rag_quick/
```

### 3. Full Benchmark

Run all methods on full dataset:

```bash
python scripts/run_rag.py \
    --dataset path/to/your/dataset.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --methods gard perm_proxy perm_logprob semantic_entropy \
    --eta 0.1 \
    --m-permutations 10 \
    --k-answers 10 \
    --output results/rag_full/
```

## Dataset Format

JSONL format with one example per line:

```json
{
  "id": "ex_000001",
  "query": "Question text...",
  "evidence": [
    {"id": "p1", "text": "passage 1 ..."},
    {"id": "p2", "text": "passage 2 ..."}
  ],
  "gold_answer": "short answer",
  "label": 0
}
```

- `label`: 0 = supported, 1 = hallucination/unsupported

Sample dataset provided: `data/sample_rag.jsonl` (20 examples)

## Implemented Modules

### Core Package (gard/)

1. **qwen_backend.py** (475 lines) ✅
   - Model loading with GPU acceleration
   - Hidden state extraction
   - Generation with sampling
   - Teacher-forced logprob computation

2. **embed.py** (180 lines) ✅
   - Extract embeddings from last hidden layer
   - Mean pooling over valid tokens
   - L2 normalization
   - Batch processing

3. **metrics.py** (240 lines) ✅
   - Geometric margin computation
   - Bivector coherence (prefix method, O(n))
   - GISR (Geometric Insufficiency Sufficiency Ratio)
   - Combined risk score

4. **perm_proxy.py** (200 lines) ✅
   - Harmonic positional weighting
   - Vectorized permutation sampling
   - Dispersion statistics (STD, MAD)

5. **perm_logprob.py** (250 lines) ✅
   - Canonical answer generation
   - Teacher-forced logprob computation
   - Batch processing across permutations
   - Dispersion statistics

6. **semantic_entropy.py** (220 lines) ✅
   - K answer generation (temperature=0.7)
   - Greedy clustering (δ=0.85)
   - Entropy computation

7. **eval.py** (320 lines) ✅
   - AUROC/AUPRC computation
   - Abstention curves
   - Spearman correlations
   - Conflict vs weakness analysis

8. **plotting.py** (280 lines) ✅
   - ROC curves
   - Precision-Recall curves
   - Abstention curves
   - Correlation scatter plots
   - Conflict/weakness breakdown

9. **prompts.py** (65 lines) ✅
   - Exact prompt template per specification
   - Evidence formatting
   - Permutation support

10. **dataset.py** (145 lines) ✅
    - JSONL loading
    - Filtering by evidence count
    - Dataset statistics

11. **utils.py** (70 lines) ✅
    - Random seed setting
    - Device management
    - Permutation generation
    - Cosine similarity

### Experiments (gard/experiments/)

12. **synthetic.py** (240 lines) ✅
    - 4 validation scenarios
    - GPU-accelerated generation
    - Theoretical validation

13. **rag.py** (350 lines) ✅
    - Full pipeline integration
    - Multi-method support
    - Batch processing
    - Result aggregation

### Scripts (scripts/)

14. **run_rag.py** (420 lines) ✅
    - CLI argument parsing
    - Pipeline orchestration
    - Progress tracking
    - Result saving (pickle + JSON)
    - Automatic plotting

15. **run_synth.py** (140 lines) ✅
    - Synthetic validation runner
    - Pass/fail reporting

### Tests (tests/)

16. **test_metrics.py** (230 lines) ✅
    - Permutation invariance
    - Coherence bounds
    - GISR properties
    - Batch processing

17. **test_perm_logprob.py** (160 lines) ✅
    - Dispersion computation
    - MAD robustness
    - Outlier handling

18. **test_semantic_entropy.py** (180 lines) ✅
    - Clustering algorithm
    - Entropy computation
    - Threshold sensitivity

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_metrics.py -v

# Run with coverage
pytest tests/ --cov=gard --cov-report=html
```

## API Usage

### Basic Example

```python
from gard.qwen_backend import QwenBackend
from gard.embed import embed_texts
from gard.metrics import compute_all_gard_metrics

# Initialize backend
backend = QwenBackend("Qwen/Qwen2.5-7B-Instruct", device="cuda")

# Your data
query = "What is the capital of France?"
evidence_texts = [
    "Paris is the capital of France.",
    "France is in Europe.",
    "The French government is based in Paris.",
]

# Extract embeddings
V = embed_texts(backend, evidence_texts)  # (n, d)
q = embed_texts(backend, [query])[0]      # (d,)

# Compute GARD metrics
metrics = compute_all_gard_metrics(V, q, eta=0.1)

print(f"GISR: {metrics['gisr'].item():.3f}")
print(f"Coherence: {metrics['C'].item():.3f}")
print(f"Risk: {metrics['risk_ga'].item():.3f}")
print(f"Decision: {'Accept' if metrics['decision_accept'] else 'Abstain'}")
```

### Permutation Baselines

```python
from gard.perm_proxy import compute_perm_proxy_dispersion
from gard.perm_logprob import compute_perm_logprob_dispersion

# Proxy dispersion
proxy_result = compute_perm_proxy_dispersion(V, q, m=10)
print(f"Proxy STD: {proxy_result['p_std']:.3f}")

# Logprob dispersion
logprob_result = compute_perm_logprob_dispersion(
    backend, query, evidence_list, m=10
)
print(f"Logprob STD: {logprob_result['lp_std']:.3f}")
```

### Semantic Entropy

```python
from gard.semantic_entropy import compute_semantic_entropy

se_result = compute_semantic_entropy(
    backend, query, evidence_list, k=10, threshold=0.85
)
print(f"Semantic Entropy: {se_result['se']:.3f}")
print(f"Clusters: {se_result['n_clusters']}")
```

## Output

Running the benchmark produces:

1. **Raw results** (pickle): Complete data for post-processing
2. **Summary** (JSON): Key metrics for each method
3. **Plots** (PNG):
   - ROC curves (AUROC)
   - Precision-Recall curves (AUPRC)
   - Abstention curves
   - Correlation scatter plots
   - Conflict vs weakness breakdown

## Performance

### Computational Complexity

- **GARD**: O(nd + d²) per example (d=hidden dim, n=evidence count)
- **Permutation Proxy**: O(mnd) per example (m=permutations)
- **Permutation Logprob**: O(m × generation) per example
- **Semantic Entropy**: O(k × generation + k²) per example

### Timing Estimates (Qwen2.5-7B on single A100)

- GARD: ~0.5s per example
- Perm Proxy (m=10): ~1s per example
- Perm Logprob (m=10): ~10s per example
- Semantic Entropy (k=10): ~15s per example

For 100 examples:
- GARD only: ~1 minute
- GARD + Proxy: ~2 minutes
- All methods: ~40 minutes

## Repository Structure

```
gard_head2head_qwen/
├── gard/                       # Core package
│   ├── __init__.py
│   ├── qwen_backend.py        # Model interface
│   ├── embed.py               # Embedding extraction
│   ├── metrics.py             # GARD metrics
│   ├── perm_proxy.py          # Proxy baseline
│   ├── perm_logprob.py        # Logprob baseline
│   ├── semantic_entropy.py    # SE baseline
│   ├── eval.py                # Evaluation metrics
│   ├── plotting.py            # Visualizations
│   ├── prompts.py             # Prompt templates
│   ├── dataset.py             # Data loading
│   ├── utils.py               # Utilities
│   └── experiments/
│       ├── synthetic.py       # Synthetic validation
│       └── rag.py             # RAG experiments
├── scripts/
│   ├── run_rag.py            # Main runner
│   └── run_synth.py          # Validation runner
├── tests/
│   ├── test_metrics.py
│   ├── test_perm_logprob.py
│   └── test_semantic_entropy.py
├── data/
│   └── sample_rag.jsonl      # Sample dataset
├── results/                   # Output directory
├── pyproject.toml
└── README.md
```

## Citation

If you use this benchmark, please cite the GARD paper:

```
[Your citation here]
```

## Implementation Notes

- **GPU Required**: Qwen2.5-7B requires ~14GB VRAM (bfloat16)
- **Batch Processing**: Optimized for throughput
- **Reproducibility**: Fixed seeds throughout
- **Extensibility**: Modular design for adding new methods

## Troubleshooting

### OOM Errors

Reduce batch sizes:
```bash
python scripts/run_rag.py ... \
    --embed-batch-size 4 \
    --logprob-batch-size 2
```

### Slow Generation

Reduce K or m:
```bash
python scripts/run_rag.py ... \
    --k-answers 5 \
    --m-permutations 5
```

### Model Loading Issues

Ensure sufficient disk space and memory. Try smaller model:
```bash
python scripts/run_rag.py \
    --model Qwen/Qwen2-0.5B ...
```

## Future Work

- [ ] Support for other LLMs (Llama, Mistral, etc.)
- [ ] Multi-GPU parallelization
- [ ] Streaming inference for long documents
- [ ] Integration with popular RAG frameworks
- [ ] Pre-computed embeddings for faster evaluation

## License

[Your license here]

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Implementation Complete**: 100% (15/15 core modules, 3 test files, 2 runners, sample data)

**Total Lines**: ~3,200 lines of production code + 570 lines of tests

**Status**: Ready for evaluation on real datasets ✅
