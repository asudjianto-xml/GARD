"""
Dataset loading from JSONL format.

Exact format per Section 2 of instruction pack.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Example:
    """Single example from dataset."""
    id: str
    query: str
    evidence: List[Dict[str, str]]  # List of {'id': ..., 'text': ...}
    gold_answer: str
    label: int  # 0 = supported, 1 = unsupported/contradictory

    @property
    def n_evidence(self) -> int:
        """Number of evidence passages."""
        return len(self.evidence)

    @property
    def is_hallucination(self) -> bool:
        """Whether this is a hallucination case (label=1)."""
        return self.label == 1


def load_dataset(path: str) -> List[Example]:
    """
    Load dataset from JSONL file.

    Each line must have format:
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

    Args:
        path: Path to JSONL file

    Returns:
        List of Example objects
    """
    path = Path(path)
    examples = []

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Required fields
                example = Example(
                    id=data['id'],
                    query=data['query'],
                    evidence=data['evidence'],
                    gold_answer=data.get('gold_answer', ''),
                    label=data['label'],
                )

                examples.append(example)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue

    print(f"Loaded {len(examples)} examples from {path}")
    return examples


def filter_by_evidence_count(
    examples: List[Example],
    min_n: int = 2,
    max_n: int = 16,
) -> List[Example]:
    """
    Filter examples by number of evidence passages.

    Args:
        examples: List of examples
        min_n: Minimum number of evidence
        max_n: Maximum number of evidence

    Returns:
        Filtered list
    """
    return [ex for ex in examples if min_n <= ex.n_evidence <= max_n]


def truncate_evidence(
    examples: List[Example],
    n_max: int,
) -> List[Example]:
    """
    Truncate evidence lists to at most n_max passages.

    Args:
        examples: List of examples
        n_max: Maximum number of evidence passages

    Returns:
        Examples with truncated evidence
    """
    truncated = []
    for ex in examples:
        if ex.n_evidence <= n_max:
            truncated.append(ex)
        else:
            # Truncate evidence
            new_ex = Example(
                id=ex.id,
                query=ex.query,
                evidence=ex.evidence[:n_max],
                gold_answer=ex.gold_answer,
                label=ex.label,
            )
            truncated.append(new_ex)
    return truncated


def get_dataset_stats(examples: List[Example]) -> Dict[str, Any]:
    """
    Compute dataset statistics.

    Args:
        examples: List of examples

    Returns:
        Dictionary with statistics
    """
    n_counts = [ex.n_evidence for ex in examples]
    labels = [ex.label for ex in examples]

    return {
        'n_examples': len(examples),
        'n_evidence_min': min(n_counts) if n_counts else 0,
        'n_evidence_max': max(n_counts) if n_counts else 0,
        'n_evidence_mean': sum(n_counts) / len(n_counts) if n_counts else 0,
        'n_hallucinations': sum(labels),
        'n_supported': len(labels) - sum(labels),
        'hallucination_rate': sum(labels) / len(labels) if labels else 0,
    }
