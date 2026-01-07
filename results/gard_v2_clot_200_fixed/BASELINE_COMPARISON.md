# Baseline Comparison: PermProxy vs PermLogprob

## Summary

Investigation revealed that the paper's reported **0.367 AUROC** came from **PermProxy** (embedding-only approximation), NOT from the proper PermLogprob baseline described in the paper text.

## Key Findings

### Two Different Implementations

| Method | Description | LLM Calls | AUROC | Source |
|--------|-------------|-----------|-------|--------|
| **PermProxy** (embedding-only) | Permutes embeddings, computes weighted dot products, applies sigmoid | ❌ No | **0.367** | `results/clot_bench_200/` |
| **PermLogprob** (proper baseline) | Generates canonical answer + computes log P(y*\|prompt_π) for 10 permutations | ✅ Yes (11 total) | **0.452** | `results/gard_v2_clot_200_fixed/` |

### Paper Description vs Implementation

**Paper text (line 303):**
> "PermProxy-MAD: Compute model answer for M=10 random permutations, measure MAD of output probabilities"

This suggests the baseline SHOULD call the LLM to compute actual model probabilities.

**Actual implementation used for 0.367 result:**
- Script: `scripts/run_rag.py` with `methods=['gard', 'perm_proxy']`
- Function: `gard/perm_proxy.py::compute_perm_proxy_dispersion()`
- Implementation: Embedding-only approximation using weighted dot products + sigmoid
- **No LLM generation**

## Detailed Comparison

### PermProxy (0.367 AUROC)

**What it does:**
```python
# 1. Permute evidence embeddings
V_permuted = permute_embeddings(V, perm_indices)  # (m, n, d)

# 2. Compute weighted alignment scores
z(π) = Σ_j w_j · (v_π(j)^T q)  # Harmonic weights

# 3. Apply sigmoid to get "probabilities"
p(π) = sigmoid(z(π))

# 4. Compute MAD of probabilities
MAD = median(|p - median(p)|)
```

**Complexity:** O(mnd) - just embedding operations

**Processing time:** ~0.5 seconds per example

**Why it's WORSE (more anti-correlated):**
- Pure embedding geometry without model context
- Harmonic weighting is arbitrary
- Sigmoid doesn't reflect actual model confidence
- More sensitive to adversarial embedding patterns

---

### PermLogprob (0.452 AUROC)

**What it does:**
```python
# 1. Generate canonical answer (greedy decoding)
y* = model.generate(query, evidence_identity)

# 2. For each permutation π:
for π in permutations:
    prompt_π = build_prompt(query, evidence_π)
    # Compute teacher-forced log probability
    lp(π) = log P(y* | prompt_π)

# 3. Compute MAD of log probabilities
MAD = median(|lp - median(lp)|)
```

**Complexity:** O(m × T_gen) - requires 11 LLM passes

**Processing time:** ~6 seconds per example

**Why it's BETTER (closer to random):**
- Uses actual model predictions
- Captures model's true uncertainty
- More robust to adversarial patterns that fool embeddings

---

## Why PermLogprob Performs Better

**Counter-intuitive result:** The "proper" baseline (0.452) is LESS anti-correlated than the approximation (0.367).

**Explanation:**

1. **PermProxy is overly sensitive** - The embedding-only approximation reacts strongly to:
   - Surface-level embedding diversity (paraphrase explosion)
   - Geometric patterns that don't reflect model behavior
   - Result: More extreme anti-correlation (0.367)

2. **PermLogprob uses model knowledge** - Actual model probabilities:
   - Incorporate semantic understanding
   - Are less sensitive to surface variation
   - Still fail on adversarial data but less dramatically
   - Result: Closer to random (0.452)

3. **Both confirm the fundamental failure:**
   - PermProxy: 0.367 < 0.5 ✅ Anti-correlated
   - PermLogprob: 0.452 < 0.5 ✅ Anti-correlated
   - Both below random, validating theory

---

## Paper Accuracy

### What the paper says:
- ✅ "PermProxy-MAD achieves 0.367 AUROC" - **Correct value**
- ❌ "Compute model answer for M=10 random permutations" - **Description misleading**
- ✅ "Below random, anti-correlated with risk" - **Correct conclusion**

### What actually happened:
- The 0.367 result used **PermProxy** (embedding approximation)
- The paper text describes **PermLogprob** (LLM generation)
- The methodology description doesn't match the implementation

---

## Recommendation

### For Paper Revision

**Option 1: Keep current results, clarify methodology**
```latex
\textbf{PermProxy-MAD}: Compute positional dispersion using an embedding-based
approximation: for M=10 random permutations, compute weighted alignment scores
z(π) = Σ_j w_j · (v_π(j)^T q) with harmonic weights, apply sigmoid to obtain
probabilities p(π) = σ(z(π)), and measure MAD.
```

**Option 2: Use proper PermLogprob baseline**
- Re-run evaluation with `perm_logprob`
- Update result to 0.452 AUROC
- Note: Still below random, conclusion unchanged
- More accurately reflects "model answer" description

### For Codebase

Both implementations are valuable:
- **PermProxy**: Fast approximation for screening (O(mnd))
- **PermLogprob**: Accurate baseline for evaluation (O(m × T_gen))

Recommend keeping both but documenting the difference clearly.

---

## Validation

All findings confirmed by:
1. ✅ Code inspection (`perm_proxy.py` vs `perm_logprob.py`)
2. ✅ Timing (0.5s vs 6s per example)
3. ✅ Results files comparison
4. ✅ Git history and evaluation scripts

The 0.367 AUROC is **reproducible and correct** for the PermProxy (embedding-only) implementation, but the paper text describing "compute model answer" is **misleading** about what was actually run.

---

## Conclusion

**Key takeaway:** The paper's 0.367 result is **correct** but came from an embedding-only approximation, not the LLM generation baseline described in the text.

**Impact:** Minimal - both baselines fail on adversarial data (AUROC < 0.5), confirming the fundamental theory. PermLogprob (0.452) is actually closer to random, making the failure slightly less dramatic but still validating the core claim.

**Action:** Clarify methodology in paper revision to accurately describe the PermProxy implementation used, or re-run with proper PermLogprob baseline and update results to 0.452.
