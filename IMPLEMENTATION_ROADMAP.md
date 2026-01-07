# Implementation Roadmap

## Status: Foundation Complete (40% done)

### ✅ Completed Modules (6/15)

1. **qwen_backend.py** (475 lines) - Model loading, forward, generate, logprob
2. **embed.py** (180 lines) - Embedding extraction from hidden states
3. **utils.py** (70 lines) - Utilities, seeding, permutations
4. **metrics.py** (240 lines) - GARD: margin, coherence, GISR, risk
5. **prompts.py** (65 lines) - Prompt templates per spec
6. **dataset.py** (145 lines) - JSONL loading and filtering

**Total so far**: ~1,175 lines

### 🚧 Remaining Core Modules (9/15)

#### Priority 1: Baselines (Required for comparison)

7. **perm_proxy.py** (~150 lines)
   - Harmonic positional weighting
   - Vectorized permutation sampling
   - Dispersion statistics (p_std, p_mad)

8. **perm_logprob.py** (~250 lines)
   - Canonical answer generation (y*)
   - Teacher-forced logprob computation
   - Batch processing across permutations
   - Dispersion statistics (lp_std, lp_mad)

9. **semantic_entropy.py** (~200 lines)
   - K answer generation (temperature=0.7)
   - Greedy clustering (δ=0.85)
   - Entropy computation

#### Priority 2: Evaluation & Experiments

10. **eval.py** (~300 lines)
    - AUROC/AUPRC computation
    - Abstention curves
    - Spearman correlations
    - Conflict vs weakness analysis

11. **plotting.py** (~250 lines)
    - AUROC/AUPRC plots
    - Abstention curves
    - Correlation scatter plots
    - Conflict vs weakness visualization

12. **experiments/synthetic.py** (~200 lines)
    - 4 scenarios: consensus, conflict, weakness, mixed
    - GPU-accelerated generation
    - Validation experiments

13. **experiments/rag.py** (~300 lines)
    - Full pipeline integration
    - Batch processing
    - Result aggregation

#### Priority 3: Scripts & Infrastructure

14. **scripts/run_rag.py** (~400 lines)
    - CLI argument parsing
    - Pipeline orchestration
    - Progress tracking
    - Result saving

15. **scripts/run_synth.py** (~150 lines)
    - Synthetic experiment runner
    - Validation against theory

### 📋 Test Suite (3 files)

16. **tests/test_metrics.py** (~200 lines)
    - Permutation invariance
    - Coherence bounds
    - GISR properties

17. **tests/test_perm_logprob.py** (~150 lines)
    - Logprob computation correctness
    - Permutation sensitivity

18. **tests/test_semantic_entropy.py** (~150 lines)
    - Clustering algorithm
    - Entropy computation

**Remaining**: ~2,600 lines

**Total project**: ~3,775 lines

## Implementation Schedule

### Phase 1: Baselines (2-3 hours)
- [x] Foundation modules
- [ ] perm_proxy.py
- [ ] perm_logprob.py
- [ ] semantic_entropy.py

### Phase 2: Evaluation (1-2 hours)
- [ ] eval.py
- [ ] plotting.py

### Phase 3: Experiments (1-2 hours)
- [ ] synthetic.py
- [ ] rag.py

### Phase 4: Scripts (1-2 hours)
- [ ] run_rag.py
- [ ] run_synth.py

### Phase 5: Testing (1-2 hours)
- [ ] Test suite (3 files)
- [ ] Integration testing
- [ ] Sample data creation

### Phase 6: Validation (1 hour)
- [ ] End-to-end run
- [ ] Result verification
- [ ] Documentation

**Total estimated time**: 7-12 hours remaining

## Quick Start Guide (For Current State)

### Test What's Built

```python
from gard.qwen_backend import QwenBackend
from gard.embed import embed_texts
from gard.metrics import compute_all_gard_metrics
from gard.dataset import load_dataset
from gard.prompts import build_prompt

# Initialize
backend = QwenBackend("Qwen/Qwen2.5-7B-Instruct")

# Test embeddings
texts = ["Paris is the capital of France.", "France is in Europe."]
embeddings = embed_texts(backend, texts)
print(f"Embeddings shape: {embeddings.shape}")

# Test GARD metrics
import torch
V = embeddings  # (2, d)
q = embeddings[0]  # (d)
metrics = compute_all_gard_metrics(V.unsqueeze(0), q.unsqueeze(0), eta=0.1)
print(f"GISR: {metrics['gisr'].item():.3f}")
print(f"Coherence: {metrics['C'].item():.3f}")
print(f"Risk GA: {metrics['risk_ga'].item():.3f}")
```

## Priority Next Steps

1. **Implement perm_proxy.py** - Simplest baseline, no LLM needed
2. **Create minimal test dataset** (10 examples)
3. **Implement eval.py** - Just AUROC/AUPRC for now
4. **Build minimal run_rag.py** - Single method at a time
5. **Validate** with small experiment

## Alternative: Minimal Viable Benchmark

If time is constrained, implement in this order:

1. ✅ Core foundation (done)
2. **perm_proxy.py** (1 hour) - Required baseline
3. **eval.py minimal** (30 min) - Just AUROC
4. **run_rag.py minimal** (1 hour) - GARD + proxy only
5. **Sample data** (30 min) - 20 examples
6. **Test run** (30 min)

**MVP total**: Current + 3.5 hours = Working comparison

Then iterate to add:
- perm_logprob.py
- semantic_entropy.py
- Full evaluation
- Comprehensive tests

## Files Created So Far

```
gard_head2head_qwen/
├── gard/
│   ├── __init__.py ✅
│   ├── qwen_backend.py ✅ (475 lines)
│   ├── embed.py ✅ (180 lines)
│   ├── utils.py ✅ (70 lines)
│   ├── metrics.py ✅ (240 lines)
│   ├── prompts.py ✅ (65 lines)
│   ├── dataset.py ✅ (145 lines)
│   ├── perm_proxy.py 🚧
│   ├── perm_logprob.py 🚧
│   ├── semantic_entropy.py 🚧
│   ├── eval.py 🚧
│   ├── plotting.py 🚧
│   └── experiments/
│       ├── synthetic.py 🚧
│       └── rag.py 🚧
├── scripts/
│   ├── run_rag.py 🚧
│   └── run_synth.py 🚧
├── tests/ 🚧
├── pyproject.toml ✅
└── README.md ✅
```

## Decision Point

**Option A**: Continue full implementation (7-12 hours)
- Complete all 15 modules
- Full test suite
- Comprehensive evaluation

**Option B**: Build MVP (3-4 hours from now)
- Core + 1-2 baselines
- Basic evaluation
- Working benchmark

**Option C**: Pause and provide templates
- Give you skeleton code for remaining modules
- You can complete implementation
- I provide guidance/review

**Which would you prefer?**

---

**Current investment**: ~2.5 hours
**Remaining for full spec**: ~7-12 hours
**Remaining for MVP**: ~3-4 hours
