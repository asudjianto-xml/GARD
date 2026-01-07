"""
Unit tests for GARD metrics.

Tests:
1. Permutation invariance (margin, GISR)
2. Coherence bounds [0, 1]
3. GISR decision consistency
4. Risk combination bounds
"""

import pytest
import torch
import numpy as np

from gard.metrics import (
    compute_margin,
    compute_coherence,
    compute_gisr,
    compute_risk_ga,
    compute_all_gard_metrics,
)


@pytest.fixture
def device():
    """Get device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_embeddings(device):
    """Create simple test embeddings."""
    d = 64
    n = 5

    # Random evidence
    V = torch.randn(n, d, device=device)
    V = V / torch.linalg.norm(V, dim=-1, keepdim=True)

    # Random query
    q = torch.randn(d, device=device)
    q = q / torch.linalg.norm(q)

    return V, q


def test_margin_computation(simple_embeddings):
    """Test margin computation."""
    V, q = simple_embeddings

    result = compute_margin(V, q)

    # Check keys
    assert "M" in result
    assert "absM" in result
    assert "s" in result

    # Check shapes
    assert result["M"].shape == ()
    assert result["absM"].shape == ()
    assert result["s"].shape == (5,)

    # Check properties
    assert torch.allclose(result["absM"], torch.abs(result["M"]))
    assert torch.allclose(result["M"], result["s"].sum())


def test_margin_permutation_invariance(device):
    """Test that margin is permutation invariant."""
    d = 64
    n = 5

    V = torch.randn(n, d, device=device)
    V = V / torch.linalg.norm(V, dim=-1, keepdim=True)

    q = torch.randn(d, device=device)
    q = q / torch.linalg.norm(q)

    # Compute margin
    result1 = compute_margin(V, q)
    M1 = result1["M"]

    # Permute evidence
    perm = torch.randperm(n)
    V_perm = V[perm]

    result2 = compute_margin(V_perm, q)
    M2 = result2["M"]

    # Margin should be identical
    assert torch.allclose(M1, M2, atol=1e-5)


def test_coherence_bounds(simple_embeddings):
    """Test that coherence is in [0, 1]."""
    V, q = simple_embeddings

    result = compute_coherence(V)

    # Check keys
    assert "C" in result
    assert "B_norm" in result
    assert "D" in result

    # Check bounds
    C = result["C"]
    assert C >= 0.0
    assert C <= 1.0


def test_coherence_aligned_vectors(device):
    """Test coherence for perfectly aligned vectors."""
    d = 64
    n = 5

    # All vectors in same direction
    base = torch.randn(d, device=device)
    base = base / torch.linalg.norm(base)

    V = base.unsqueeze(0).repeat(n, 1)
    V = V + 0.01 * torch.randn_like(V)
    V = V / torch.linalg.norm(V, dim=-1, keepdim=True)

    result = compute_coherence(V)
    C = result["C"]

    # For aligned vectors, coherence should be very low
    assert C < 0.1


def test_coherence_orthogonal_vectors(device):
    """Test coherence for orthogonal vectors."""
    d = 64

    # Two orthogonal vectors
    v1 = torch.zeros(d, device=device)
    v1[0] = 1.0

    v2 = torch.zeros(d, device=device)
    v2[1] = 1.0

    V = torch.stack([v1, v2])

    result = compute_coherence(V)
    C = result["C"]

    # For 2 orthogonal vectors, coherence should be high (near 1)
    assert C > 0.8


def test_gisr_computation(simple_embeddings):
    """Test GISR computation."""
    V, q = simple_embeddings
    eta = 0.1

    result = compute_gisr(V, q, eta)

    # Check keys
    assert "gisr" in result
    assert "tau" in result
    assert "L_Q" in result
    assert "decision_accept" in result

    # Check shapes
    assert result["gisr"].shape == ()
    assert result["tau"].shape == ()

    # Check properties
    n = V.shape[0]
    L_Q = result["L_Q"]
    tau = result["tau"]
    gisr = result["gisr"]

    # Verify tau = n * L_Q * eta
    assert torch.allclose(tau, n * L_Q * eta, atol=1e-5)

    # Decision should match GISR >= 1
    expected_decision = gisr >= 1.0
    assert result["decision_accept"] == expected_decision


def test_gisr_permutation_invariance(device):
    """Test that GISR is permutation invariant."""
    d = 64
    n = 5
    eta = 0.1

    V = torch.randn(n, d, device=device)
    V = V / torch.linalg.norm(V, dim=-1, keepdim=True)

    q = torch.randn(d, device=device)
    q = q / torch.linalg.norm(q)

    # Compute GISR
    result1 = compute_gisr(V, q, eta)
    gisr1 = result1["gisr"]

    # Permute evidence
    perm = torch.randperm(n)
    V_perm = V[perm]

    result2 = compute_gisr(V_perm, q, eta)
    gisr2 = result2["gisr"]

    # GISR should be identical
    assert torch.allclose(gisr1, gisr2, atol=1e-5)


def test_risk_ga_computation(simple_embeddings):
    """Test risk_GA computation."""
    V, q = simple_embeddings
    eta = 0.1
    alpha = 0.5

    result = compute_risk_ga(V, q, eta, alpha)

    # Check that all metrics are present
    assert "M" in result
    assert "C" in result
    assert "gisr" in result
    assert "risk_ga" in result

    # Check risk formula
    gisr = result["gisr"]
    C = result["C"]
    risk = result["risk_ga"]

    expected_risk = alpha * (1.0 / (gisr + 1e-12)) + (1 - alpha) * C

    assert torch.allclose(risk, expected_risk, atol=1e-5)


def test_risk_bounds(simple_embeddings):
    """Test that risk is non-negative."""
    V, q = simple_embeddings
    eta = 0.1
    alpha = 0.5

    result = compute_risk_ga(V, q, eta, alpha)
    risk = result["risk_ga"]

    assert risk >= 0.0


def test_all_gard_metrics(simple_embeddings):
    """Test compute_all_gard_metrics convenience function."""
    V, q = simple_embeddings
    eta = 0.1
    alpha = 0.5

    result = compute_all_gard_metrics(V, q, eta, alpha)

    # Check that all expected metrics are present
    expected_keys = [
        "M", "absM", "s",
        "C", "B_norm", "D",
        "gisr", "tau", "L_Q", "decision_accept",
        "risk_ga",
    ]

    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_batch_processing(device):
    """Test batch processing of multiple examples."""
    d = 64
    n = 5
    B = 3  # Batch size

    # Create batch of embeddings
    V = torch.randn(B, n, d, device=device)
    V = V / torch.linalg.norm(V, dim=-1, keepdim=True)

    q = torch.randn(B, d, device=device)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)

    eta = 0.1

    result = compute_all_gard_metrics(V, q, eta)

    # Check batch dimensions
    assert result["M"].shape == (B,)
    assert result["C"].shape == (B,)
    assert result["gisr"].shape == (B,)
    assert result["risk_ga"].shape == (B,)


def test_single_vs_batch(device):
    """Test that single and batch modes give same results."""
    d = 64
    n = 5
    eta = 0.1

    V = torch.randn(n, d, device=device)
    V = V / torch.linalg.norm(V, dim=-1, keepdim=True)

    q = torch.randn(d, device=device)
    q = q / torch.linalg.norm(q)

    # Single mode
    result_single = compute_all_gard_metrics(V, q, eta)

    # Batch mode (batch size 1)
    V_batch = V.unsqueeze(0)
    q_batch = q.unsqueeze(0)
    result_batch = compute_all_gard_metrics(V_batch, q_batch, eta)

    # Results should match
    assert torch.allclose(result_single["M"], result_batch["M"][0], atol=1e-5)
    assert torch.allclose(result_single["C"], result_batch["C"][0], atol=1e-5)
    assert torch.allclose(result_single["gisr"], result_batch["gisr"][0], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
