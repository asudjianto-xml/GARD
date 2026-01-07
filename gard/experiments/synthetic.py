"""
Synthetic validation experiments.

4 scenarios to validate GARD metrics:
1. Consensus: All evidence aligned, low coherence, high GISR → accept
2. Conflict: Evidence contradictory, high coherence → abstain
3. Weakness: Evidence insufficient, low GISR → abstain
4. Mixed: Combination of above

GPU-accelerated generation and validation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SyntheticScenario:
    """Single synthetic scenario."""
    name: str
    n_evidence: int
    query_direction: torch.Tensor  # (d,)
    evidence_directions: List[torch.Tensor]  # List of (d,)
    expected_decision: str  # "accept" or "abstain"
    scenario_type: str  # "consensus", "conflict", "weakness", "mixed"


def create_synthetic_scenarios(
    d: int = 768,
    device: torch.device = None,
) -> List[SyntheticScenario]:
    """
    Create 4 synthetic validation scenarios.

    Args:
        d: Embedding dimension
        device: Device for tensors

    Returns:
        List of synthetic scenarios
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scenarios = []

    # Scenario 1: Consensus (all aligned with query)
    # Expected: Low coherence, high GISR → accept
    q1 = torch.randn(d, device=device)
    q1 = q1 / torch.linalg.norm(q1)

    # All evidence aligned in same direction with small noise
    v1_list = []
    for _ in range(5):
        v = q1 + 0.1 * torch.randn(d, device=device)
        v = v / torch.linalg.norm(v)
        v1_list.append(v)

    scenarios.append(SyntheticScenario(
        name="Consensus",
        n_evidence=5,
        query_direction=q1,
        evidence_directions=v1_list,
        expected_decision="accept",
        scenario_type="consensus",
    ))

    # Scenario 2: Conflict (contradictory evidence)
    # Expected: High coherence → abstain
    q2 = torch.randn(d, device=device)
    q2 = q2 / torch.linalg.norm(q2)

    # Half evidence points in +q direction, half in -q direction
    v2_list = []
    for i in range(6):
        if i < 3:
            v = q2 + 0.1 * torch.randn(d, device=device)
        else:
            v = -q2 + 0.1 * torch.randn(d, device=device)
        v = v / torch.linalg.norm(v)
        v2_list.append(v)

    scenarios.append(SyntheticScenario(
        name="Conflict",
        n_evidence=6,
        query_direction=q2,
        evidence_directions=v2_list,
        expected_decision="abstain",
        scenario_type="conflict",
    ))

    # Scenario 3: Weakness (insufficient evidence)
    # Expected: Low GISR → abstain
    q3 = torch.randn(d, device=device)
    q3 = q3 / torch.linalg.norm(q3)

    # Few evidence with weak alignment (orthogonal)
    v3_list = []
    for _ in range(2):
        v = torch.randn(d, device=device)
        v = v / torch.linalg.norm(v)
        # Make orthogonal to query
        v = v - (v @ q3) * q3
        v = v / torch.linalg.norm(v)
        v3_list.append(v)

    scenarios.append(SyntheticScenario(
        name="Weakness",
        n_evidence=2,
        query_direction=q3,
        evidence_directions=v3_list,
        expected_decision="abstain",
        scenario_type="weakness",
    ))

    # Scenario 4: Mixed (some aligned, some conflicting, some weak)
    # Expected: Moderate coherence, moderate GISR → borderline
    q4 = torch.randn(d, device=device)
    q4 = q4 / torch.linalg.norm(q4)

    v4_list = []
    # 2 aligned
    for _ in range(2):
        v = q4 + 0.1 * torch.randn(d, device=device)
        v = v / torch.linalg.norm(v)
        v4_list.append(v)

    # 1 conflicting
    v = -q4 + 0.1 * torch.randn(d, device=device)
    v = v / torch.linalg.norm(v)
    v4_list.append(v)

    # 1 weak (orthogonal)
    v = torch.randn(d, device=device)
    v = v / torch.linalg.norm(v)
    v = v - (v @ q4) * q4
    v = v / torch.linalg.norm(v)
    v4_list.append(v)

    scenarios.append(SyntheticScenario(
        name="Mixed",
        n_evidence=4,
        query_direction=q4,
        evidence_directions=v4_list,
        expected_decision="abstain",
        scenario_type="mixed",
    ))

    return scenarios


def validate_scenario(
    scenario: SyntheticScenario,
    eta: float = 0.1,
    alpha: float = 0.5,
) -> Dict[str, any]:
    """
    Validate a single synthetic scenario.

    Args:
        scenario: Synthetic scenario
        eta: Perturbation radius for GISR
        alpha: Weight for risk combination

    Returns:
        Dictionary with validation results
    """
    from ..metrics import compute_all_gard_metrics

    # Stack evidence embeddings
    V = torch.stack(scenario.evidence_directions)  # (n, d)
    q = scenario.query_direction  # (d,)

    # Compute all GARD metrics
    metrics = compute_all_gard_metrics(V, q, eta=eta, alpha=alpha)

    # Make decision
    gisr = metrics["gisr"].item()
    decision = "accept" if gisr >= 1.0 else "abstain"

    # Check if matches expected
    matches_expected = (decision == scenario.expected_decision)

    return {
        "scenario": scenario.name,
        "type": scenario.scenario_type,
        "n_evidence": scenario.n_evidence,
        "M": metrics["M"].item(),
        "absM": metrics["absM"].item(),
        "C": metrics["C"].item(),
        "gisr": gisr,
        "risk_ga": metrics["risk_ga"].item(),
        "decision": decision,
        "expected_decision": scenario.expected_decision,
        "matches_expected": matches_expected,
    }


def run_synthetic_validation(
    d: int = 768,
    eta: float = 0.1,
    alpha: float = 0.5,
    device: torch.device = None,
) -> Dict[str, any]:
    """
    Run full synthetic validation experiment.

    Args:
        d: Embedding dimension
        eta: Perturbation radius
        alpha: Weight for risk combination
        device: Device for computation

    Returns:
        Dictionary with all validation results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating synthetic scenarios...")
    scenarios = create_synthetic_scenarios(d=d, device=device)

    print(f"Validating {len(scenarios)} scenarios...")
    results = []
    for scenario in scenarios:
        result = validate_scenario(scenario, eta=eta, alpha=alpha)
        results.append(result)

        print(f"  {result['scenario']:12s}: "
              f"GISR={result['gisr']:.3f}, "
              f"C={result['C']:.3f}, "
              f"Decision={result['decision']:7s}, "
              f"Expected={result['expected_decision']:7s}, "
              f"Match={'✓' if result['matches_expected'] else '✗'}")

    # Summary statistics
    n_matches = sum(r["matches_expected"] for r in results)
    accuracy = n_matches / len(results)

    summary = {
        "n_scenarios": len(results),
        "n_matches": n_matches,
        "accuracy": accuracy,
        "results": results,
    }

    print(f"\nValidation Accuracy: {accuracy:.1%} ({n_matches}/{len(results)})")

    return summary
