"""
Unit tests for semantic entropy.

Tests:
1. Greedy clustering algorithm
2. Entropy computation
3. Threshold sensitivity
"""

import pytest
import torch
import numpy as np

from gard.semantic_entropy import (
    greedy_clustering,
    compute_entropy_from_clusters,
)


@pytest.fixture
def device():
    """Get device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def identical_embeddings(device):
    """Create identical embeddings (should form 1 cluster)."""
    d = 64
    k = 5

    # All same embedding
    base = torch.randn(d, device=device)
    base = base / torch.linalg.norm(base)

    embeddings = base.unsqueeze(0).repeat(k, 1)

    return embeddings


@pytest.fixture
def orthogonal_embeddings(device):
    """Create orthogonal embeddings (should form k clusters)."""
    k = 5
    d = 64

    # Create orthogonal basis
    embeddings = torch.eye(k, d, device=device)

    return embeddings


@pytest.fixture
def two_cluster_embeddings(device):
    """Create embeddings that should form 2 clusters."""
    d = 64

    # Cluster 1: 3 similar embeddings
    base1 = torch.randn(d, device=device)
    base1 = base1 / torch.linalg.norm(base1)

    cluster1 = base1.unsqueeze(0).repeat(3, 1)
    cluster1 = cluster1 + 0.05 * torch.randn_like(cluster1)
    cluster1 = cluster1 / torch.linalg.norm(cluster1, dim=-1, keepdim=True)

    # Cluster 2: 2 similar embeddings (orthogonal to cluster 1)
    base2 = torch.randn(d, device=device)
    base2 = base2 - (base2 @ base1) * base1
    base2 = base2 / torch.linalg.norm(base2)

    cluster2 = base2.unsqueeze(0).repeat(2, 1)
    cluster2 = cluster2 + 0.05 * torch.randn_like(cluster2)
    cluster2 = cluster2 / torch.linalg.norm(cluster2, dim=-1, keepdim=True)

    embeddings = torch.cat([cluster1, cluster2], dim=0)

    return embeddings


def test_clustering_identical(identical_embeddings):
    """Test clustering with identical embeddings."""
    threshold = 0.85

    clusters, centroids = greedy_clustering(identical_embeddings, threshold)

    # Should form 1 cluster
    assert len(clusters) == 1
    assert len(clusters[0]) == 5

    # Centroid shape
    assert centroids.shape == (1, 64)


def test_clustering_orthogonal(orthogonal_embeddings):
    """Test clustering with orthogonal embeddings."""
    threshold = 0.85

    clusters, centroids = greedy_clustering(orthogonal_embeddings, threshold)

    # Should form k clusters (one per embedding)
    assert len(clusters) == 5
    for i, cluster in enumerate(clusters):
        assert len(cluster) == 1
        assert cluster[0] == i

    # Centroid shape
    assert centroids.shape == (5, 64)


def test_clustering_two_groups(two_cluster_embeddings):
    """Test clustering with two groups."""
    threshold = 0.85

    clusters, centroids = greedy_clustering(two_cluster_embeddings, threshold)

    # Should form 2-3 clusters (depending on noise)
    assert 2 <= len(clusters) <= 3

    # Total elements should be 5
    total_elements = sum(len(c) for c in clusters)
    assert total_elements == 5

    # Centroid shape should match number of clusters
    assert centroids.shape == (len(clusters), 64)


def test_clustering_threshold_sensitivity(two_cluster_embeddings):
    """Test that threshold affects clustering."""
    # High threshold (more clusters)
    clusters_high, _ = greedy_clustering(two_cluster_embeddings, threshold=0.99)

    # Low threshold (fewer clusters)
    clusters_low, _ = greedy_clustering(two_cluster_embeddings, threshold=0.50)

    # Higher threshold should produce more clusters
    assert len(clusters_high) >= len(clusters_low)


def test_entropy_single_cluster():
    """Test entropy with single cluster (zero entropy)."""
    clusters = [[0, 1, 2, 3, 4]]
    k = 5

    entropy = compute_entropy_from_clusters(clusters, k)

    # Entropy should be zero (no uncertainty)
    assert entropy < 1e-6


def test_entropy_all_separate():
    """Test entropy with all separate clusters (maximum entropy)."""
    clusters = [[0], [1], [2], [3], [4]]
    k = 5

    entropy = compute_entropy_from_clusters(clusters, k)

    # Entropy should be log(k)
    expected_entropy = np.log(k)
    assert abs(entropy - expected_entropy) < 0.01


def test_entropy_two_clusters():
    """Test entropy with two equal clusters."""
    clusters = [[0, 1, 2], [3, 4, 5]]
    k = 6

    entropy = compute_entropy_from_clusters(clusters, k)

    # Entropy should be log(2)
    expected_entropy = np.log(2)
    assert abs(entropy - expected_entropy) < 0.01


def test_entropy_bounds():
    """Test that entropy is non-negative."""
    clusters = [[0, 1], [2, 3, 4]]
    k = 5

    entropy = compute_entropy_from_clusters(clusters, k)

    # Entropy should be non-negative
    assert entropy >= 0


def test_entropy_ordering():
    """Test that more clusters means higher entropy."""
    k = 10

    # 1 cluster
    clusters_1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    entropy_1 = compute_entropy_from_clusters(clusters_1, k)

    # 2 clusters
    clusters_2 = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    entropy_2 = compute_entropy_from_clusters(clusters_2, k)

    # 5 clusters
    clusters_5 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    entropy_5 = compute_entropy_from_clusters(clusters_5, k)

    # More clusters = higher entropy
    assert entropy_1 < entropy_2 < entropy_5


def test_clustering_first_element():
    """Test that first element always forms first cluster."""
    d = 64
    k = 5
    device = torch.device("cpu")

    embeddings = torch.randn(k, d, device=device)
    embeddings = embeddings / torch.linalg.norm(embeddings, dim=-1, keepdim=True)

    clusters, _ = greedy_clustering(embeddings, threshold=0.85)

    # First cluster should contain element 0
    assert 0 in clusters[0]


def test_centroid_normalization(two_cluster_embeddings):
    """Test that centroids are normalized."""
    threshold = 0.85

    clusters, centroids = greedy_clustering(two_cluster_embeddings, threshold)

    # Check that all centroids have unit norm
    norms = torch.linalg.norm(centroids, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
