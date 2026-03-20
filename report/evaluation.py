"""
Evaluation utilities for document ranking.
Implements NDCG@k and other IR metrics.
"""

import math
from typing import List, Dict, Tuple


def dcg_at_k(scores: List[float], k: int) -> float:
    """Compute DCG@k."""
    dcg = 0.0
    for i in range(min(k, len(scores))):
        dcg += (2 ** scores[i] - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_indices: List[int], relevance: Dict[int, int], k: int = 10) -> float:
    """
    Compute NDCG@k.

    Args:
        ranked_indices: List of document indices in ranked order.
        relevance: Dict mapping document index -> relevance grade.
        k: Cutoff.

    Returns:
        NDCG@k score (0.0 to 1.0).
    """
    # Actual DCG
    actual_scores = []
    for i in range(min(k, len(ranked_indices))):
        idx = ranked_indices[i]
        actual_scores.append(relevance.get(idx, 0))
    actual_dcg = dcg_at_k(actual_scores, k)

    # Ideal DCG
    ideal_scores = sorted(relevance.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(ranked_indices: List[int], relevant_set: set, k: int = 10) -> float:
    """Compute Precision@k."""
    hits = sum(1 for idx in ranked_indices[:k] if idx in relevant_set)
    return hits / k


def recall_at_k(ranked_indices: List[int], relevant_set: set, k: int = 10) -> float:
    """Compute Recall@k."""
    if not relevant_set:
        return 0.0
    hits = sum(1 for idx in ranked_indices[:k] if idx in relevant_set)
    return hits / len(relevant_set)


def mrr(ranked_indices: List[int], relevant_set: set) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, idx in enumerate(ranked_indices):
        if idx in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranking(ranked_indices: List[int], relevance: Dict[int, int], k: int = 10) -> Dict[str, float]:
    """
    Compute a full suite of evaluation metrics.

    Args:
        ranked_indices: Document indices in ranked order.
        relevance: Dict mapping doc_index -> relevance grade (0,1,2,3).
        k: Cutoff for metrics.

    Returns:
        Dict with metric names and scores.
    """
    relevant_set = {idx for idx, rel in relevance.items() if rel > 0}

    return {
        f"NDCG@{k}": ndcg_at_k(ranked_indices, relevance, k),
        f"Precision@{k}": precision_at_k(ranked_indices, relevant_set, k),
        f"Recall@{k}": recall_at_k(ranked_indices, relevant_set, k),
        "MRR": mrr(ranked_indices, relevant_set),
    }
