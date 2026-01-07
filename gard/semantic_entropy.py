"""
Semantic Entropy baseline.

Exact implementation per Section 5.1 of instruction pack.

Reference: Kuhn et al. "Semantic Uncertainty: Linguistic Invariances for
Uncertainty Estimation in Natural Language Generation"

Key points:
1. Generate K answers at temperature=0.7
2. Greedy clustering with cosine similarity threshold δ=0.85
3. Compute entropy over cluster sizes
"""

import torch
from typing import Dict, List, Tuple
from .utils import cosine_similarity
from .embed import embed_texts


def generate_multiple_answers(
    backend,
    prompt: str,
    k: int = 10,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """
    Generate K diverse answers using sampling.

    Args:
        backend: QwenBackend instance
        prompt: Input prompt
        k: Number of answers to generate
        max_new_tokens: Maximum tokens per answer
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        List of K answer strings
    """
    answers = []
    for _ in range(k):
        # Generate one answer at a time to ensure diversity
        batch_answers = backend.generate(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        answers.append(batch_answers[0].strip())

    return answers


def greedy_clustering(
    embeddings: torch.Tensor,
    threshold: float = 0.85,
) -> Tuple[List[List[int]], torch.Tensor]:
    """
    Greedy clustering based on cosine similarity.

    Algorithm:
    1. Start with first answer as first cluster
    2. For each subsequent answer:
       - Compute similarity to all cluster centroids
       - If max similarity >= threshold, add to that cluster
       - Otherwise, create new cluster

    Args:
        embeddings: Answer embeddings (K, d)
        threshold: Cosine similarity threshold δ

    Returns:
        - clusters: List of lists of answer indices
        - centroids: Cluster centroids (n_clusters, d)
    """
    K = embeddings.shape[0]
    device = embeddings.device

    # Initialize with first answer as first cluster
    clusters = [[0]]
    centroids = [embeddings[0]]

    # Process remaining answers
    for k in range(1, K):
        emb = embeddings[k]  # (d,)

        # Compute similarities to all centroids
        sims = []
        for centroid in centroids:
            sim = cosine_similarity(emb, centroid)
            sims.append(sim.item())

        # Find most similar cluster
        max_sim = max(sims)
        max_idx = sims.index(max_sim)

        if max_sim >= threshold:
            # Add to existing cluster
            clusters[max_idx].append(k)

            # Update centroid (mean of all embeddings in cluster)
            cluster_indices = clusters[max_idx]
            cluster_embs = embeddings[cluster_indices]
            new_centroid = cluster_embs.mean(dim=0)
            new_centroid = new_centroid / torch.linalg.norm(new_centroid)
            centroids[max_idx] = new_centroid
        else:
            # Create new cluster
            clusters.append([k])
            centroids.append(emb)

    # Convert centroids to tensor
    centroids_tensor = torch.stack(centroids)  # (n_clusters, d)

    return clusters, centroids_tensor


def compute_entropy_from_clusters(
    clusters: List[List[int]],
    k: int,
    eps: float = 1e-12,
) -> float:
    """
    Compute entropy from cluster assignments.

    H = -Σ_c (|c|/K) * log(|c|/K)

    Args:
        clusters: List of clusters (each cluster is list of indices)
        k: Total number of answers
        eps: Small epsilon for numerical stability

    Returns:
        Entropy value
    """
    # Compute cluster probabilities
    probs = []
    for cluster in clusters:
        prob = len(cluster) / k
        probs.append(prob)

    # Compute entropy
    entropy = 0.0
    for p in probs:
        if p > eps:
            entropy -= p * torch.log(torch.tensor(p)).item()

    return entropy


def compute_semantic_entropy(
    backend,
    query: str,
    evidence: List[Dict[str, str]],
    k: int = 10,
    threshold: float = 0.85,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 8,
) -> Dict[str, any]:
    """
    Compute semantic entropy baseline.

    Full pipeline per Section 5.1:
    1. Generate K answers at temperature=0.7
    2. Embed all answers
    3. Greedy clustering with threshold δ=0.85
    4. Compute entropy over cluster sizes

    Args:
        backend: QwenBackend instance
        query: Question string
        evidence: List of evidence dicts
        k: Number of answers to generate
        threshold: Clustering similarity threshold δ
        max_new_tokens: Maximum tokens per answer
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        batch_size: Batch size for embedding

    Returns:
        Dictionary with:
        - se: Semantic entropy
        - n_clusters: Number of clusters
        - clusters: List of clusters
        - answers: Generated answers
        - embeddings: Answer embeddings (K, d)
    """
    from .prompts import build_prompt

    # Build prompt
    prompt = build_prompt(query, evidence)

    # Generate K answers
    answers = generate_multiple_answers(
        backend,
        prompt,
        k=k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Embed all answers
    embeddings = embed_texts(
        backend,
        answers,
        batch_size=batch_size,
        max_len=256,
    )  # (K, d)

    # Greedy clustering
    clusters, centroids = greedy_clustering(embeddings, threshold)

    # Compute entropy
    entropy = compute_entropy_from_clusters(clusters, k)

    return {
        "se": entropy,
        "n_clusters": len(clusters),
        "clusters": clusters,
        "answers": answers,
        "embeddings": embeddings,
    }


def compute_semantic_entropy_batch(
    backend,
    queries: List[str],
    evidence_lists: List[List[Dict[str, str]]],
    k: int = 10,
    threshold: float = 0.85,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 8,
) -> List[Dict[str, any]]:
    """
    Compute semantic entropy for batch of examples.

    Args:
        backend: QwenBackend instance
        queries: List of question strings
        evidence_lists: List of evidence lists
        k: Number of answers per example
        threshold: Clustering similarity threshold
        max_new_tokens: Maximum tokens per answer
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        batch_size: Batch size for embedding

    Returns:
        List of result dictionaries, one per example
    """
    results = []
    for query, evidence in zip(queries, evidence_lists):
        result = compute_semantic_entropy(
            backend,
            query,
            evidence,
            k=k,
            threshold=threshold,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        results.append(result)

    return results
