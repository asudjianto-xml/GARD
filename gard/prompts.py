"""
Prompt templates for Qwen2.5 evaluation.

Exact template per Section 5.2 Step A of instruction pack.
"""

from typing import List, Dict


PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the evidence.
If the evidence is insufficient or contradictory, output exactly: INSUFFICIENT.

Question:
{query}

Evidence:
{evidence_list}

Answer:"""


def format_evidence_list(evidence: List[Dict[str, str]]) -> str:
    """
    Format evidence list for prompt.

    Args:
        evidence: List of dicts with 'id' and 'text' keys

    Returns:
        Formatted evidence string
    """
    lines = []
    for i, ev in enumerate(evidence, 1):
        lines.append(f"[{i}] {ev['text']}")
    return "\n".join(lines)


def build_prompt(query: str, evidence: List[Dict[str, str]]) -> str:
    """
    Build full prompt from query and evidence.

    Args:
        query: Question string
        evidence: List of evidence dicts

    Returns:
        Formatted prompt string
    """
    evidence_str = format_evidence_list(evidence)
    return PROMPT_TEMPLATE.format(
        query=query,
        evidence_list=evidence_str,
    )


def build_prompt_permuted(query: str, evidence: List[Dict[str, str]], permutation: List[int]) -> str:
    """
    Build prompt with permuted evidence order.

    Args:
        query: Question string
        evidence: List of evidence dicts
        permutation: List of indices defining order

    Returns:
        Formatted prompt with permuted evidence
    """
    permuted_evidence = [evidence[i] for i in permutation]
    return build_prompt(query, permuted_evidence)
