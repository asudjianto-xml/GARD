"""
Embedding extraction from Qwen2.5 hidden states.

Exact implementation per Section 3 of instruction pack.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from .qwen_backend import QwenBackend


def embed_texts(
    backend: QwenBackend,
    texts: List[str],
    batch_size: int = 8,
    max_len: int = 256,
) -> torch.Tensor:
    """
    Extract embeddings from texts using internal hidden states.

    Implementation per Section 3 specification:
    - Use last hidden layer
    - Exclude padding tokens
    - Exclude special tokens (BOS, EOS)
    - Mean pool over valid tokens
    - Normalize to unit norm

    Args:
        backend: QwenBackend instance
        texts: List of text strings
        batch_size: Batch size for processing
        max_len: Maximum sequence length

    Returns:
        Embeddings (N, d) float32, L2-normalized
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        inputs = backend.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        ).to(backend.device)

        # Forward pass
        with torch.no_grad():
            outputs = backend.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        # Extract last hidden layer
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        # Get special token IDs
        special_ids = backend.get_special_token_ids()
        bos_id = special_ids['bos']
        eos_id = special_ids['eos']

        # Process each sequence in batch
        batch_embeddings = []
        for j in range(hidden.shape[0]):
            # Get tokens and mask for this sequence
            token_ids = inputs['input_ids'][j]
            attention_mask = inputs['attention_mask'][j]

            # Build valid token mask
            # Exclude padding (attention_mask == 0)
            valid_mask = attention_mask.bool()

            # Exclude BOS token if present
            if bos_id is not None:
                valid_mask = valid_mask & (token_ids != bos_id)

            # Exclude EOS token if present
            if eos_id is not None:
                valid_mask = valid_mask & (token_ids != eos_id)

            # Extract valid hidden states
            valid_hidden = hidden[j, valid_mask, :]  # (num_valid, hidden_dim)

            if valid_hidden.shape[0] == 0:
                # No valid tokens - use all non-padding
                valid_mask = attention_mask.bool()
                valid_hidden = hidden[j, valid_mask, :]

            # Mean pool
            embedding = valid_hidden.mean(dim=0)  # (hidden_dim,)

            # Normalize
            norm = torch.linalg.norm(embedding)
            if norm > 1e-12:
                embedding = embedding / norm
            else:
                embedding = embedding / 1e-12

            batch_embeddings.append(embedding)

        # Stack batch
        batch_emb = torch.stack(batch_embeddings)  # (batch, hidden_dim)
        all_embeddings.append(batch_emb.cpu().float())

    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)  # (N, hidden_dim)

    return embeddings


def embed_query(
    backend: QwenBackend,
    query: str,
    max_len: int = 256,
) -> torch.Tensor:
    """
    Embed a single query.

    Args:
        backend: QwenBackend instance
        query: Query string
        max_len: Maximum sequence length

    Returns:
        Embedding (d,) float32, L2-normalized
    """
    emb = embed_texts(backend, [query], batch_size=1, max_len=max_len)
    return emb.squeeze(0)


def embed_evidence(
    backend: QwenBackend,
    evidence_list: List[str],
    batch_size: int = 8,
    max_len: int = 256,
) -> torch.Tensor:
    """
    Embed a list of evidence passages.

    Args:
        backend: QwenBackend instance
        evidence_list: List of evidence strings
        batch_size: Batch size for processing
        max_len: Maximum sequence length

    Returns:
        Embeddings (n, d) float32, L2-normalized
    """
    return embed_texts(backend, evidence_list, batch_size=batch_size, max_len=max_len)


def embed_answers(
    backend: QwenBackend,
    answers: List[str],
    batch_size: int = 16,
    max_len: int = 128,
) -> torch.Tensor:
    """
    Embed answer strings (for semantic entropy clustering).

    Args:
        backend: QwenBackend instance
        answers: List of answer strings
        batch_size: Batch size for processing
        max_len: Maximum sequence length

    Returns:
        Embeddings (K, d) float32, L2-normalized
    """
    return embed_texts(backend, answers, batch_size=batch_size, max_len=max_len)
