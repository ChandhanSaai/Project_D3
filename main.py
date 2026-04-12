"""
D3: Debate, Deliberate, Decide
A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation

Paper: https://arxiv.org/abs/2410.04663
"""

import argparse
from config import D3Config
from protocols.more import MOREProtocol
from protocols.samre import SAMREProtocol


def main():
    parser = argparse.ArgumentParser(
        description="D3: Cost-Aware Adversarial Framework for LLM Evaluation"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["more", "samre"],
        default="more",
        help="Evaluation protocol: 'more' (Multi-Advocate One-Round) or 'samre' (Single-Advocate Multi-Round)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mt-bench", "alignbench", "auto-j"],
        default="mt-bench",
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--num-advocates",
        type=int,
        default=3,
        help="Number of advocate agents (k)",
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
        default="gpt-4",
        help="LLM model to use for agents",
    )
    parser.add_argument(
        "--use-jury",
        action="store_true",
        help="Enable optional jury for final decision",
    )

    args = parser.parse_args()

    config = D3Config(
        model=args.model,
        num_advocates=args.num_advocates,
        token_budget=args.token_budget,
        use_jury=args.use_jury,
        dataset=args.dataset,
    )

    if args.protocol == "more":
        protocol = MOREProtocol(config)
    else:
        protocol = SAMREProtocol(config)

    print(f"Running D3 with {args.protocol.upper()} protocol on {args.dataset}")
    print(f"Advocates: {args.num_advocates} | Token budget: {args.token_budget}")
    print(f"Model: {args.model} | Jury: {'enabled' if args.use_jury else 'disabled'}")
    print("-" * 60)

    results = protocol.run()
    print("\nEvaluation complete.")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
