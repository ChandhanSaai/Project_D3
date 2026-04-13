"""
Evaluation metrics for the D3 framework.

Implements the metrics reported in the paper (Section 4, Tables 2-5):

  Agreement Rate  — % of samples where D3 verdict matches human preference
  Position Bias   — % of samples where swapping answer order flips the verdict
  Cohen's Kappa   — inter-rater agreement between D3 and human labels
  Cost Efficiency — average tokens per evaluation, broken down by protocol

Reference: Section 4.2, Section 5 of the D3 paper (arXiv:2410.04663)
"""

import logging
import math

logger = logging.getLogger(__name__)


def agreement_rate(predictions, human_labels):
    """Compute agreement rate between D3 verdicts and human labels.

    Only counts samples where human_label != 0 (i.e. non-ties with a
    clear human preference).  D3 ties (prediction=0) against a clear
    human preference count as disagreement.

    Args:
        predictions: list[int] — D3 winner per sample (1, 2, or 0).
        human_labels: list[int] — human preference per sample (1, 2, or 0).

    Returns:
        dict with:
            agreement  — float in [0, 1]
            total      — number of comparable samples
            matched    — number that agreed
    """
    if len(predictions) != len(human_labels):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(human_labels)} human labels."
        )

    matched = 0
    total = 0

    for pred, label in zip(predictions, human_labels):
        if label == 0:
            # No clear human preference — skip
            continue
        total += 1
        if pred == label:
            matched += 1

    rate = matched / total if total > 0 else 0.0

    return {
        "agreement": rate,
        "total": total,
        "matched": matched,
    }


def position_bias(results_original, results_swapped):
    """Measure position bias by comparing verdicts before/after answer swap.

    For each sample, run D3 with (answer1, answer2) and again with
    (answer2, answer1).  If the winner flips to follow the *position*
    rather than the *content*, that's a position-biased sample.

    A verdict is position-biased if:
        original winner = 1  AND  swapped winner = 1  (always picks position 1)
     or original winner = 2  AND  swapped winner = 2  (always picks position 2)

    A verdict is consistent if:
        original winner = 1  AND  swapped winner = 2  (tracks the content)
     or original winner = 2  AND  swapped winner = 1

    Ties in either run are excluded from the count.

    Args:
        results_original: list[int] — winners from original order.
        results_swapped:  list[int] — winners from swapped order.

    Returns:
        dict with:
            bias_rate   — fraction of comparable pairs that are biased
            consistent  — count of consistent verdicts
            biased      — count of biased verdicts
            total       — count of comparable pairs (both non-tie)
    """
    if len(results_original) != len(results_swapped):
        raise ValueError("Length mismatch between original and swapped results.")

    consistent = 0
    biased = 0
    total = 0

    for orig, swap in zip(results_original, results_swapped):
        if orig == 0 or swap == 0:
            continue  # skip ties
        total += 1
        # Content-consistent: orig=1,swap=2 or orig=2,swap=1
        if orig != swap:
            consistent += 1
        else:
            # Same position wins regardless of content → biased
            biased += 1

    bias_rate = biased / total if total > 0 else 0.0

    return {
        "bias_rate": bias_rate,
        "consistent": consistent,
        "biased": biased,
        "total": total,
    }


def cohens_kappa(predictions, human_labels):
    """Compute Cohen's Kappa between D3 verdicts and human labels.

    Measures agreement beyond chance.  Uses labels {1, 2} only
    (ties are excluded from both sides).

    Interpretation (Landis & Koch, 1977):
        < 0.00   Poor
        0.00-0.20  Slight
        0.21-0.40  Fair
        0.41-0.60  Moderate
        0.61-0.80  Substantial
        0.81-1.00  Almost perfect

    Args:
        predictions: list[int] — D3 verdicts.
        human_labels: list[int] — human preferences.

    Returns:
        dict with:
            kappa       — Cohen's Kappa score
            p_observed  — observed agreement
            p_expected  — expected agreement by chance
            n           — number of compared pairs
    """
    if len(predictions) != len(human_labels):
        raise ValueError("Length mismatch.")

    # Filter to pairs where both are non-tie
    pairs = [
        (p, h) for p, h in zip(predictions, human_labels)
        if p != 0 and h != 0
    ]
    n = len(pairs)

    if n == 0:
        return {"kappa": 0.0, "p_observed": 0.0, "p_expected": 0.0, "n": 0}

    # Observed agreement
    agree = sum(1 for p, h in pairs if p == h)
    p_o = agree / n

    # Expected agreement by chance
    # P(both say 1) + P(both say 2)
    pred_1 = sum(1 for p, _ in pairs if p == 1) / n
    pred_2 = sum(1 for p, _ in pairs if p == 2) / n
    human_1 = sum(1 for _, h in pairs if h == 1) / n
    human_2 = sum(1 for _, h in pairs if h == 2) / n
    p_e = (pred_1 * human_1) + (pred_2 * human_2)

    if abs(1.0 - p_e) < 1e-10:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "kappa": kappa,
        "p_observed": p_o,
        "p_expected": p_e,
        "n": n,
    }


def cost_summary(token_counts):
    """Summarise token costs across a batch of evaluations.

    Args:
        token_counts: list[int] — total_tokens from each protocol.run() result.

    Returns:
        dict with:
            total_tokens — sum
            mean_tokens  — average per evaluation
            min_tokens   — cheapest single evaluation
            max_tokens   — most expensive single evaluation
            std_tokens   — standard deviation
            count        — number of evaluations
    """
    if not token_counts:
        return {
            "total_tokens": 0,
            "mean_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "std_tokens": 0,
            "count": 0,
        }

    n = len(token_counts)
    total = sum(token_counts)
    mean = total / n
    min_t = min(token_counts)
    max_t = max(token_counts)

    if n > 1:
        variance = sum((t - mean) ** 2 for t in token_counts) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return {
        "total_tokens": total,
        "mean_tokens": mean,
        "min_tokens": min_t,
        "max_tokens": max_t,
        "std_tokens": std,
        "count": n,
    }


def print_report(agreement, bias, kappa, cost):
    """Pretty-print a full evaluation report to stdout.

    Args:
        agreement: dict from agreement_rate()
        bias: dict from position_bias() (or None if not measured)
        kappa: dict from cohens_kappa()
        cost: dict from cost_summary()
    """
    print("=" * 60)
    print("D3 Evaluation Report")
    print("=" * 60)

    print(f"\nAgreement with human labels:")
    print(f"  Rate:        {agreement['agreement']:.1%}")
    print(f"  Matched:     {agreement['matched']} / {agreement['total']}")

    print(f"\nCohen's Kappa:")
    print(f"  Kappa:       {kappa['kappa']:.4f}")
    print(f"  P(observed): {kappa['p_observed']:.4f}")
    print(f"  P(expected): {kappa['p_expected']:.4f}")
    print(f"  Pairs:       {kappa['n']}")

    if bias is not None:
        print(f"\nPosition Bias:")
        print(f"  Bias rate:   {bias['bias_rate']:.1%}")
        print(f"  Consistent:  {bias['consistent']} / {bias['total']}")
        print(f"  Biased:      {bias['biased']} / {bias['total']}")

    print(f"\nCost:")
    print(f"  Total tokens: {cost['total_tokens']:,}")
    print(f"  Mean/eval:    {cost['mean_tokens']:,.0f}")
    print(f"  Min/Max:      {cost['min_tokens']:,} / {cost['max_tokens']:,}")
    print(f"  Std dev:      {cost['std_tokens']:,.0f}")
    print(f"  Evaluations:  {cost['count']}")

    print()
    print("=" * 60)
