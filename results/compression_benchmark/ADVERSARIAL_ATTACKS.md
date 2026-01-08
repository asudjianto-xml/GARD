# Adversarial Attack Examples from CLOT Bench

This document explains the three adversarial attack families in CLOT Bench and shows why they fool hallucination detection systems.

---

## Attack 1: Balanced Contradiction

**Strategy**: Include equal amounts of evidence supporting contradictory answers.

### Example 1: Darwin's Birth Year

**Question**: "In what year was Darwin born?"

**Evidence** (4 say 1809, 4 say 1810):
1. "According to official records, Biographies consistently state Darwin was born in **1810**."
2. "Darwin was born in **1810** according to historical records."
3. "Biographies consistently state Darwin was born in **1809**."
4. "Birth records confirm Darwin was born in **1809**."
5. "Historical documentation places Darwin's birth in **1809**."
6. "Historical documentation places Darwin's birth in **1810**."
7. "The year **1809** marks the birth of Darwin."
8. "Biographies consistently state Darwin was born in **1810**."

**Ground Truth**: Darwin was born in **1809** (correct answer is in the evidence)

**The Trap**:
- Model sees 50/50 split between two years
- Both sides use authoritative language ("official records", "biographies")
- Model either:
  - **Picks one arbitrarily** → 50% chance of hallucination
  - **Says "insufficient"** → Fails to answer despite correct info being present

**Model's Response**:
> "INSUFFICIENT. The evidence provided contains conflicting information about the year Darwin was born, with some sources stating 1810 and others stating 1809."

**Detection Results**:
- **GARD**: ❌ Accepts (C=0.377, low conflict detected)
- **Compression**: ✅ Abstains (ISR=0.578, detected uncertainty)

**Why It Fools GARD's GISR**: The balanced contradiction creates a **moderate margin** toward one answer (whichever the model leans toward). GISR=8.438 suggests robustness, but it's actually risky.

---

## Attack 2: Paraphrase Explosion

**Strategy**: Repeat the same (wrong) answer 10+ times using different wording.

### Example: Smartphone Invention

**Question**: "When was the smartphone invented?"

**Evidence** (all say 2007, but rephrased 10 times):
1. "The smartphone was invented in **2007**."
2. "In the year **2007**, the smartphone was created."
3. "The year **2007 CE** saw the invention of the smartphone."
4. "**2007** is when the smartphone was first developed."
5. "The smartphone came into existence in the year **2007**."
6. "Development of the smartphone occurred in **2007 CE**."
7. "The smartphone was first introduced in **2007**."
8. "The year **2007** marked the debut of the smartphone."
9. "The smartphone originated in **2007 CE**."
10. "Creation of the smartphone took place in **2007**."

**Ground Truth**: Depends on definition - first smartphone concepts existed in 1992 (IBM Simon), iPhone in 2007. This example is marked **SAFE** (if considering iPhone as "the smartphone").

**The Trap**:
- Model sees overwhelming consensus (10/10 agreement)
- Surface-level diversity (different phrasings) masks repetition
- Exploits:
  - **Permutation-based methods**: Low dispersion (all say same thing) → High confidence
  - **Margin-based methods**: All vectors align → Large margin → False robustness

**Why This Fools Compression Metrics**:
- **Low LP-MAD**: Permuting evidence gives similar probabilities (all support 2007)
- **High ISR**: Low dispersion → High information budget → "Sufficient" to answer
- **Low RoH**: Appears low-risk due to consistency

**Why This Can Fool GARD's GISR**:
- All embeddings point in similar direction → **Large margin**
- **High GISR** suggests robustness, but it's just repetition
- This is why GISR has **inverted correlation (0.2457 AUROC)** on adversarial data

---

## Attack 3: Hybrid Trap (Refute)

**Strategy**: Include authoritative-sounding wrong answer + vague correct info + explicit refutation.

### Example: Capital of Italy

**Question**: "What is the capital of Italy?"

**Evidence**:
1. "According to the **Comprehensive National Gazetteer**, the capital of Italy is **Milan**, which serves as the administrative and political center."
2. "Italy's government is located in **Rome**." *(vague correct)*
3. "Official government records confirm that **Rome is in Italy**." *(true but unhelpful)*
4. "In the **Definitive Statistical Yearbook**, **Milan** is listed as the capital of Italy."
5. "The **Comprehensive Geographic Authority** confirms that **Milan** is designated as the capital of Italy in official government records."
6. "Contrary to some sources, **Milan is not the capital of Italy**." *(explicit refutation)*

**Ground Truth**: Rome is the capital (correct)

**The Trap**:
- **3 strong authoritative claims** for Milan (with fake source names)
- **1 vague correct mention** of Rome (but doesn't say it's the capital)
- **1 explicit refutation** of Milan (but buried at the end)
- Model faces conflicting signals:
  - Authority and specificity point to Milan
  - Vague mentions of Rome
  - Last-minute refutation creates confusion

**Model's Response**:
> "INSUFFICIENT. The evidence provided is contradictory, with some sources stating that Milan is the capital and others confirming that Rome is the capital of Italy."

**Detection Results**:
- **GARD**: ✅ Abstains (C=0.479, detected conflict)
- **Compression**: ❌ Accepts (ISR=3.235, low dispersion)

**Why Different Methods Disagree**:
- **GARD's Coherence** detects geometric conflict between "Milan" and "Rome" vectors
- **Compression's ISR** sees low permutation dispersion (most evidence points to Milan consistently)

---

## Summary: How Each Attack Exploits Detection Systems

| Attack Type | Exploits | Fools GARD? | Fools Compression? |
|-------------|----------|-------------|-------------------|
| **Balanced Contradiction** | Margin-based confidence (model picks one side) | Sometimes (if conflict not geometric) | Sometimes (if dispersion low) |
| **Paraphrase Explosion** | Repetition = consensus, low dispersion | ✅ Yes (high margin, high GISR) | ✅ Yes (low dispersion, high ISR) |
| **Hybrid Trap** | Authority bias, vague correct info | Sometimes (depends on geometric conflict) | ✅ Yes (consistent wrong majority) |

---

## Why GARD's GISR Has Inverted Correlation (0.2457 AUROC)

**The Problem**: Adversarial attacks specifically **target margin**:

1. **Paraphrase Explosion**:
   - All evidence vectors align → **Large margin**
   - High GISR suggests robustness → Model confidently answers
   - **But answer can be wrong!** (if all 10 paraphrases are wrong)

2. **Hybrid Trap**:
   - Strong authoritative claims align → **Large margin** toward wrong answer
   - High GISR → Model confident
   - **But wrong!**

3. **Balanced Contradiction** (when model picks):
   - Model commits to one side → **Moderate margin**
   - GISR suggests moderate robustness
   - **50% chance of being wrong**

**Insight**: Adversarial evidence is crafted to be "**confidently wrong**" by creating coherent alignment toward incorrect answers.

---

## Why Compression Metrics Are Anti-Correlated (As Predicted)

Chlon et al. 2025 predicted this: **Permutation-based uncertainty fails because**:

1. **Paraphrase Explosion**:
   - Permuting 10 identical paraphrases → **Low dispersion**
   - ISR suggests sufficient information → Appears safe
   - **But all 10 could be wrong!**

2. **Hybrid Trap**:
   - Most evidence (5/6) says Milan → **Low dispersion**
   - Permutations don't change the majority → High ISR
   - **Wrong majority = hallucination**

3. **Surface consistency ≠ semantic correctness**:
   - Low LP-MAD just means evidence is consistent
   - Doesn't mean evidence is **true**

**Result**: All compression metrics (ISR 0.475, RoH 0.491, LP-MAD 0.452) below random, confirming theory.

---

## What Works: GARD's Coherence (C)

**GARD Coherence (0.5827 AUROC)** succeeds because:

1. **Detects geometric conflict** via bivector wedge products:
   - Balanced Contradiction: ✅ C increases when "1809" ∧ "1810" vectors conflict
   - Hybrid Trap: ✅ C increases when "Rome" ∧ "Milan" vectors conflict

2. **Not fooled by repetition**:
   - Paraphrase Explosion: 10 identical claims still produce low C (no conflict)
   - **But**: Can be caught by coverage threshold (GARD abstains if C < C_hi)

3. **Geometric disagreement is direct evidence of problems**:
   - If evidence vectors point in different directions → **Real conflict**
   - Not based on dispersion or margin → Harder to fool

---

## Real Example Metrics

### Balanced Contradiction (Darwin)
- **GARD**: C=0.377 (lowish conflict), GISR=8.438 (high margin) → **Accept** ❌
- **Compression**: ISR=0.578 (insufficient info), LP-MAD=3.25 (high dispersion) → **Abstain** ✅

### Hybrid Trap (Italy Capital)
- **GARD**: C=0.479 (detected conflict) → **Abstain** ✅
- **Compression**: ISR=3.235 (sufficient info), LP-MAD=0.75 (low dispersion) → **Accept** ❌

### Paraphrase Explosion (WWW invented)
- **GARD**: C=0.369 (low conflict), GISR=8.25 (high margin) → **Accept** (can be risky)
- **Compression**: ISR=0.770 (insufficient), LP-MAD=2.25 → **Abstain** ✅

---

## Takeaways

1. **No single method is perfect** - each attack exploits different assumptions
2. **GARD Coherence (C) is most robust** (0.5827 AUROC) but not perfect
3. **GISR needs recalibration** - high margin can mean "confidently wrong"
4. **Compression metrics fail as predicted** - permutation-based uncertainty is exploitable
5. **Ensemble approach recommended**: Use C for conflict detection + ISR for information sufficiency

The adversarial attacks demonstrate that:
- **Consensus ≠ Truth** (Paraphrase Explosion)
- **Confidence ≠ Correctness** (High margin can be wrong)
- **Authority ≠ Accuracy** (Hybrid Trap's fake sources)

Robust hallucination detection must go beyond surface-level patterns.
