"""
D3: Debate, Deliberate, Decide
A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation

Paper: https://arxiv.org/abs/2410.04663
GitHub: https://github.com/abirharrasse/D3-Judge

Usage:
    python main.py --protocol more --question "..." --answer1 "..." --answer2 "..."
    python main.py --protocol samre --question "..." --answer1 "..." --answer2 "..."
"""

import argparse
import json
import logging
import sys

from config import D3Config
from protocols.more import MOREProtocol
from protocols.samre import SAMREProtocol


def setup_logging(verbose):
    """Configure logging level."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="D3: Cost-Aware Adversarial Framework for LLM Evaluation"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["more", "samre"],
        default="more",
        help="Evaluation protocol: 'more' (Multi-Advocate One-Round) or "
             "'samre' (Single-Advocate Multi-Round)",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The question/prompt to evaluate answers for",
    )
    parser.add_argument(
        "--answer1",
        type=str,
        required=True,
        help="Candidate answer 1",
    )
    parser.add_argument(
        "--answer2",
        type=str,
        required=True,
        help="Candidate answer 2",
    )
    parser.add_argument(
        "--num-advocates",
        type=int,
        default=3,
        help="Number of advocate agents per answer (k) [MORE only]",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum debate rounds [SAMRE only]",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=4096,
        help="Maximum token budget for evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-nano",
        help="LLM model to use for agents",
    )
    parser.add_argument(
        "--use-jury",
        action="store_true",
        help="Enable jury panel for final decision (default: judge only)",
    )
    parser.add_argument(
        "--no-jury",
        action="store_true",
        help="Disable jury, use judge scores only",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.05,
        help="Score gap stability threshold for SAMRE stopping",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write full results as JSON",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("d3")

    # Build config
    use_jury = True
    if args.no_jury:
        use_jury = False
    elif args.use_jury:
        use_jury = True

    config = D3Config(
        model=args.model,
        num_advocates=args.num_advocates,
        max_rounds=args.max_rounds,
        token_budget=args.token_budget,
        convergence_threshold=args.convergence_threshold,
        use_jury=use_jury,
    )

    # Select protocol
    if args.protocol == "more":
        protocol = MOREProtocol(config)
    else:
        protocol = SAMREProtocol(config)

    # Header
    print("=" * 60)
    print("D3: Debate, Deliberate, Decide")
    print("=" * 60)
    print(f"Protocol:   {args.protocol.upper()}")
    print(f"Model:      {args.model}")
    print(f"Jury:       {'enabled' if use_jury else 'disabled'}")
    if args.protocol == "more":
        print(f"Advocates:  {args.num_advocates} per answer")
    else:
        print(f"Max rounds: {args.max_rounds}")
        print(f"Budget:     {args.token_budget} tokens")
    print("-" * 60)

    # Run evaluation
    results = protocol.run(
        question=args.question,
        answer1=args.answer1,
        answer2=args.answer2,
    )

    # Display results
    print()
    # Check for protocol-level errors (e.g. all advocates failed)
    if results.get("error"):
        print(f"ERROR: {results['error']}")
        print(f"Total tokens used: {results['total_tokens']}")
        print()
        print("=" * 60)
        return results

    winner = results["winner"]
    if winner == 1:
        print("WINNER: Answer 1")
    elif winner == 2:
        print("WINNER: Answer 2")
    else:
        print("RESULT: Tie")

    if args.protocol == "more":
        print(f"Judge scores: {results['scores']}")
    else:
        print(f"Rounds completed: {results['rounds_completed']}")
        print(f"Stop reason: {results['stop_reason']}")
        print(f"Scores per round: {results['scores_per_round']}")

    print(f"Total tokens: {results['total_tokens']}")

    if results.get("verdict"):
        v = results["verdict"]
        print(f"Jury vote counts: {v['vote_counts']}")
        print(f"Majority reached: {v['majority_reached']}")
        if v["tie_broken"]:
            print("(Winner decided by Judge tie-break)")

    # Optionally write full results to JSON
    if args.output_json:
        # Make results JSON-serializable (remove non-serializable items)
        json_results = {
            "winner": results["winner"],
            "protocol": results["protocol"],
            "total_tokens": results["total_tokens"],
        }

        if args.protocol == "more":
            json_results["scores"] = results["scores"]
        else:
            json_results["rounds_completed"] = results["rounds_completed"]
            json_results["stop_reason"] = results["stop_reason"]
            json_results["scores_per_round"] = results["scores_per_round"]

        if results.get("verdict"):
            json_results["verdict"] = {
                "winner": results["verdict"]["winner"],
                "vote_counts": results["verdict"]["vote_counts"],
                "majority_reached": results["verdict"]["majority_reached"],
                "tie_broken": results["verdict"]["tie_broken"],
                "rationales": results["verdict"]["rationales"],
            }

        with open(args.output_json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nFull results written to: {args.output_json}")

    print()
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
