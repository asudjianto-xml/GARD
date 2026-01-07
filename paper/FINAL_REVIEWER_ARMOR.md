# Final Reviewer-Armor Edits

## Context
Paper was already strong. These are **minimal, surgical wording changes** to make the AUROC framing impossible to misinterpret.

---

## Changes Made (3 precise edits)

### 1. **Table Caption Enhanced** (Line 317)

**Before:**
```latex
\caption{Performance on CLOT Bench (200 examples: 80 Balanced Contradiction,
60 Hybrid Trap, 60 Paraphrase). GARD exposes a Pareto frontier via $C_{\text{hi}}$;
PermProxy-MAD operates at a single fixed point.}
```

**After:**
```latex
\caption{Safety–coverage tradeoff on CLOT Bench (200 adversarial examples:
80 Balanced Contradiction, 60 Hybrid Trap, 60 Paraphrase).
GARD is a risk-controlled abstention system: conflict recall and coverage are
primary metrics. AUROC is reported for completeness but does not reflect safety
constraint satisfaction. GARD exposes a Pareto frontier via $C_{\text{hi}}$;
PermProxy-MAD operates at a single fixed point.}
```

**Why**: Primes reader BEFORE they see numbers. Sets correct evaluation frame.

---

### 2. **Pre-Table Sentence Added** (Line 336)

**Added immediately before "Table~\ref{tab:main_results} reveals...":**
```latex
Because GARD enforces hard safety constraints rather than ranking all examples,
scalar ranking metrics such as AUROC are insufficient to characterize performance
and may obscure safety–coverage tradeoffs.
```

**Why**: Makes it impossible for reviewer to default to "AUROC leaderboard thinking."

---

### 3. **Footnote Added** (Line 313)

**Added at first mention of AUROC:**
```latex
AUROC\footnote{AUROC assumes a single monotone tradeoff between true and false
positive rates. Selective prediction systems with abstention regions intentionally
violate this assumption; see \cite{geifman2017selective}.} measures ranking quality...
```

**Why**: Anchors evaluation choice in established literature (selective prediction). Gives reviewers a citation trail if they want to verify.

**Reference (already in bibliography, lines 553-556):**
```bibtex
Geifman, Y., & El-Yaniv, R. (2017).
Selective Classification for Deep Neural Networks.
NeurIPS, Vol. 30, pp. 4878--4887.
```

---

## Impact

### Before these edits:
- Paper made correct argument
- Reviewer might still mentally default to AUROC comparison
- Would require careful reading to avoid misinterpretation

### After these edits:
- **Impossible to read Table 1 without absorbing the correct frame first**
- Three reinforcements (paragraph, caption, footnote) before numbers appear
- Reviewer would have to actively ignore explicit statements to misinterpret

---

## What Was NOT Changed

✅ **No methodological changes**
✅ **No new experiments**
✅ **No table restructuring** (already optimal from previous revision)
✅ **No changes to theory section**

Only **wording-level precision** to lock in the correct interpretation.

---

## Reviewer Response Scenarios

### Scenario 1: "Why is GARD's AUROC flat?"
**Paper now says explicitly (line 313):**
> "The flat AUROC across GARD operating points (0.583) reflects that varying C_hi
> shifts the decision boundary without changing the underlying geometric ordering—
> **this is correct by design**."

### Scenario 2: "PermProxy has lower AUROC but seems reasonable?"
**Paper now says explicitly (line 336):**
> "Because GARD enforces hard safety constraints rather than ranking all examples,
> scalar ranking metrics such as AUROC are **insufficient to characterize performance**."

**Plus (line 410):**
> "PermProxy is a **single dominated point**... dominated by GARD at C_hi=0.42
> (91.7% paraphrase, 55.7% conflict)."

### Scenario 3: "This seems like redefining metrics to favor your method"
**Paper now says explicitly (caption + footnote):**
> "GARD is a risk-controlled abstention system... AUROC is reported for completeness
> but does not reflect safety constraint satisfaction."

**Plus citation to Geifman 2017**, showing this is standard practice in selective prediction.

---

## Three-Layer Defense

1. **Footnote** (line 313): Academic justification via citation
2. **Paragraph** (line 312-313): Philosophical argument
3. **Caption** (line 317): Operational instruction ("these are the metrics that matter")
4. **Pre-table sentence** (line 336): Final warning before numbers

Reviewer encounters **4 explicit statements** before seeing AUROC column.

---

## Analogies (For Understanding Why This Works)

This is exactly like:

### Conformal Prediction
- **Metric that matters**: Coverage guarantee (e.g., 90%)
- **Metric that doesn't**: Prediction interval width on individual examples
- GARD's C_hi ≈ Conformal's α (risk budget)

### Neyman-Pearson Testing
- **Metric that matters**: Type I error rate (α, safety constraint)
- **Metric that doesn't**: Type II error rate (power, varies by choice)
- You **choose α first**, then report power

### GARD
- **Metric that matters**: Conflict recall (safety constraint)
- **Metric that doesn't**: AUROC (varies by C_hi, reflects ordering not safety)
- You **choose C_hi first** (application risk tolerance), then report coverage

---

## Bottom Line

**Paper was already correct.**
**These edits make it impossible to read incorrectly.**

The argument is now:
1. Front-loaded (caption)
2. Justified (footnote → citation)
3. Explained (paragraph)
4. Reinforced (pre-table sentence)

A reviewer would have to **deliberately ignore 4 explicit statements** to misinterpret AUROC.

---

## Files Updated

**Single file:**
- `/home/asudjianto/jupyterlab/gard_head2head_qwen/paper/gard_revised.tex`

**Lines changed:** 3 precise edits (313, 317, 336)

**Total new words:** ~80 words across all edits

**Structural changes:** 0

---

## Status: Ready for Submission

This paper is now **maximally reviewer-proof** on the AUROC interpretation issue.

The only remaining tasks (if desired):
1. Compile LaTeX to check formatting
2. Verify figure quality (300 DPI for camera-ready)
3. Proofread for typos
4. Add author contributions / acknowledgments

**No further methodological work needed.**
