# GARD Paper Reference

## Official Paper

The complete paper for this work is available on SSRN:

**GARD: Geometric Abstention via Robust Disagreement - A Geometric Algebra Framework for Hallucination Detection in Large Language Models**

📄 **SSRN Link:** https://hq.ssrn.com/submissions/MyPapers.cfm?partid=7085335

---

## Abstract

Hallucination detection in large language models (LLMs) faces a fundamental tension: permutation-based uncertainty estimation methods optimize expected correctness under the model distribution but fail catastrophically on adversarial inputs, while existing geometric approaches conflate evidence disagreement with insufficient support. We introduce **GARD** (*Geometric Abstention via Robust Disagreement*), a framework grounded in Geometric Algebra that *separates* conflict detection from weakness detection via orthogonal geometric observables.

## Key Results

- **Controllable Pareto Frontier:** GARD exposes safety-coverage tradeoffs ranging from 96.4% conflict recall (safety-first) to 100% paraphrase acceptance (coverage-first)
- **PermProxy Failure Validated:** Permutation methods achieve 0.367 AUROC (below random, anti-correlated with risk)
- **Deterministic O(N) Computation:** Linear-time with worst-case robustness guarantees

## Citation

Please cite the SSRN paper when referencing this work.

---

## Repository Contents

This repository contains:
- Implementation of GARD v2 with two-gate decision rule
- CLOT Bench adversarial evaluation dataset
- Experimental results and analysis
- Comparison with permutation-based baselines

For the complete theoretical framework, proofs, and detailed experimental methodology, please refer to the SSRN paper linked above.
