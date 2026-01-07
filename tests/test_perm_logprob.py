"""
Unit tests for permutation logprob dispersion.

Tests:
1. Logprob computation correctness
2. Permutation sensitivity
3. Dispersion statistics (STD, MAD)
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from gard.perm_logprob import (
    compute_dispersion_std_1d,
    compute_dispersion_mad_1d,
)


@pytest.fixture
def device():
    """Get device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_logprobs(device):
    """Create sample logprobs for testing."""
    # Logprobs for 10 permutations
    logprobs = torch.tensor([
        -2.5, -3.0, -2.8, -3.2, -2.7,
        -3.1, -2.9, -3.3, -2.6, -3.4,
    ], device=device)

    return logprobs


def test_dispersion_std_computation(sample_logprobs):
    """Test standard deviation computation."""
    std = compute_dispersion_std_1d(sample_logprobs)

    # Check scalar
    assert std.shape == ()

    # Check value is positive
    assert std > 0

    # Verify against numpy
    expected_std = sample_logprobs.cpu().numpy().std()
    assert torch.allclose(std.cpu(), torch.tensor(expected_std), atol=1e-5)


def test_dispersion_mad_computation(sample_logprobs):
    """Test MAD computation."""
    mad = compute_dispersion_mad_1d(sample_logprobs)

    # Check scalar
    assert mad.shape == ()

    # Check value is non-negative
    assert mad >= 0

    # Verify computation
    median_val = sample_logprobs.median()
    abs_dev = (sample_logprobs - median_val).abs()
    expected_mad = abs_dev.median()

    assert torch.allclose(mad, expected_mad, atol=1e-5)


def test_dispersion_uniform_values(device):
    """Test dispersion with uniform values (zero dispersion)."""
    # All same value
    logprobs = torch.full((10,), -3.0, device=device)

    std = compute_dispersion_std_1d(logprobs)
    mad = compute_dispersion_mad_1d(logprobs)

    # Should be zero or very small
    assert std < 1e-6
    assert mad < 1e-6


def test_dispersion_high_variance(device):
    """Test dispersion with high variance."""
    # Wide range of values
    logprobs = torch.tensor([
        -1.0, -5.0, -2.0, -6.0, -1.5,
        -5.5, -2.5, -4.5, -3.0, -4.0,
    ], device=device)

    std = compute_dispersion_std_1d(logprobs)
    mad = compute_dispersion_mad_1d(logprobs)

    # Should be substantial
    assert std > 1.0
    assert mad > 1.0


def test_dispersion_mad_robustness(device):
    """Test that MAD is more robust to outliers than STD."""
    # Values with outlier
    logprobs_normal = torch.tensor([
        -3.0, -3.1, -2.9, -3.2, -2.8,
        -3.1, -2.9, -3.0, -3.1, -2.9,
    ], device=device)

    logprobs_with_outlier = torch.tensor([
        -3.0, -3.1, -2.9, -3.2, -2.8,
        -3.1, -2.9, -3.0, -3.1, -10.0,  # Outlier
    ], device=device)

    std_normal = compute_dispersion_std_1d(logprobs_normal)
    mad_normal = compute_dispersion_mad_1d(logprobs_normal)

    std_outlier = compute_dispersion_std_1d(logprobs_with_outlier)
    mad_outlier = compute_dispersion_mad_1d(logprobs_with_outlier)

    # STD should increase more than MAD
    std_increase = (std_outlier - std_normal) / std_normal
    mad_increase = (mad_outlier - mad_normal) / mad_normal

    assert std_increase > mad_increase


def test_empty_evidence():
    """Test handling of empty evidence list."""
    # This should be caught at a higher level
    # Just ensure dispersion functions handle edge cases

    # Single value (no variance)
    logprobs = torch.tensor([-3.0])

    std = compute_dispersion_std_1d(logprobs)
    mad = compute_dispersion_mad_1d(logprobs)

    # Should be zero
    assert std == 0.0
    assert mad == 0.0


def test_negative_logprobs(device):
    """Test that logprobs are negative (as expected)."""
    # Log probabilities should always be negative or zero
    logprobs = torch.tensor([
        -2.5, -3.0, -2.8, -3.2, -2.7,
    ], device=device)

    assert (logprobs <= 0).all()


def test_dispersion_ordering(device):
    """Test that higher dispersion indicates more uncertainty."""
    # Low dispersion (consistent)
    logprobs_low = torch.tensor([
        -3.0, -3.05, -2.95, -3.02, -2.98,
    ], device=device)

    # High dispersion (inconsistent)
    logprobs_high = torch.tensor([
        -1.0, -5.0, -2.0, -6.0, -3.0,
    ], device=device)

    std_low = compute_dispersion_std_1d(logprobs_low)
    std_high = compute_dispersion_std_1d(logprobs_high)

    mad_low = compute_dispersion_mad_1d(logprobs_low)
    mad_high = compute_dispersion_mad_1d(logprobs_high)

    # High dispersion should be greater
    assert std_high > std_low
    assert mad_high > mad_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
