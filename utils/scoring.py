"""
Scoring utilities for D3 Framework.

Implements probabilistic score gap analysis showing that
aggregating across k advocates increases expected score separation.
"""

import math


def compute_score_gap(scores_a, scores_b):
    """Compute the score gap between two sets of evaluation scores.

    Args:
        scores_a: Scores for response A.
        scores_b: Scores for response B.

    Returns:
        float: The score gap (positive means A is better).
    """
    mean_a = sum(scores_a) / len(scores_a) if scores_a else 0
    mean_b = sum(scores_b) / len(scores_b) if scores_b else 0
    return mean_a - mean_b


def aggregate_scores(advocate_scores):
    """Aggregate scores from multiple advocates.

    Per the D3 paper, aggregating across k advocates provably
    increases expected score separation.

    Args:
        advocate_scores: List of scores from k advocates.

    Returns:
        dict with 'mean', 'std', and 'confidence'.
    """
    if not advocate_scores:
        return {"mean": 0, "std": 0, "confidence": 0}

    k = len(advocate_scores)
    mean = sum(advocate_scores) / k

    if k > 1:
        variance = sum((s - mean) ** 2 for s in advocate_scores) / (k - 1)
        std = math.sqrt(variance)
    else:
        std = 0

    # Confidence increases with sqrt(k) advocates
    confidence = 1 - (std / math.sqrt(k)) if k > 0 and std > 0 else 1.0

    return {"mean": mean, "std": std, "confidence": min(max(confidence, 0), 1)}
