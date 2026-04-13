"""
Batch evaluation runner for the D3 framework.

Runs a D3 protocol (MORE or SAMRE) over an entire benchmark dataset,
collects per-sample results, and computes aggregate metrics.

Optionally measures position bias by re-running each sample with
swapped answer order (Section 5.3 of the paper).

Usage:
    from config import D3Config
    from evaluation.runner import BatchRunner

    config = D3Config(model="gpt-5.4-nano", protocol="more")
    runner = BatchRunner(config)
    report = runner.run("mt-bench")
    report = runner.run("mt-bench", measure_bias=True)

Reference: Section 4 (Experimental Setup), Section 5 (Results)
           of the D3 paper (arXiv:2410.04663)
"""

import json
import logging
import os
import time

from config import D3Config
from evaluation.benchmarks import BenchmarkLoader
from evaluation import metrics
from protocols.more import MOREProtocol
from protocols.samre import SAMREProtocol

logger = logging.getLogger(__name__)


class BatchRunner:
    """Run D3 evaluation over a full benchmark dataset."""

    def __init__(self, config, protocol_name="more"):
        """
        Args:
            config: D3Config instance.
            protocol_name: "more" or "samre".
        """
        self.config = config
        self.protocol_name = protocol_name.lower()

        if self.protocol_name not in ("more", "samre"):
            raise ValueError(
                f"Unknown protocol: {protocol_name}. Use 'more' or 'samre'."
            )

    def _create_protocol(self):
        """Create a fresh protocol instance."""
        if self.protocol_name == "more":
            return MOREProtocol(self.config)
        else:
            return SAMREProtocol(self.config)

    def run(self, dataset_name, data_path=None, measure_bias=False,
            max_samples=None, output_path=None):
        """Run evaluation on the full benchmark.

        Args:
            dataset_name: "mt-bench", "alignbench", or "auto-j".
            data_path: Custom path to dataset file (optional).
            measure_bias: If True, re-run each sample with swapped
                          answer order to measure position bias.
            max_samples: Limit to first N samples (for debugging/cost).
            output_path: If set, write per-sample results as JSON.

        Returns:
            dict with:
                results        — list of per-sample result dicts
                agreement      — agreement_rate() output
                kappa          — cohens_kappa() output
                bias           — position_bias() output (or None)
                cost           — cost_summary() output
                metadata       — run configuration info
        """
        # Load dataset
        loader = BenchmarkLoader(dataset_name)
        samples = loader.load(data_path)

        if max_samples is not None and max_samples > 0:
            samples = samples[:max_samples]
            logger.info("Limited to %d samples.", max_samples)

        total = len(samples)
        logger.info(
            "Starting batch evaluation: %s protocol, %d samples from %s",
            self.protocol_name.upper(), total, dataset_name,
        )

        # Collect results
        per_sample = []
        predictions = []
        human_labels = []
        token_counts = []

        # For position bias (swapped runs)
        swapped_predictions = [] if measure_bias else None

        start_time = time.time()

        for i, sample in enumerate(samples):
            logger.info(
                "Sample %d/%d (id=%s)", i + 1, total,
                sample.get("question_id", i),
            )

            # --- Original order ---
            protocol = self._create_protocol()
            result = protocol.run(
                question=sample["question"],
                answer1=sample["answer1"],
                answer2=sample["answer2"],
            )

            winner = result["winner"]
            tokens = result["total_tokens"]

            predictions.append(winner)
            human_labels.append(sample["human_label"])
            token_counts.append(tokens)

            sample_record = {
                "index": i,
                "question_id": sample.get("question_id", i),
                "prediction": winner,
                "human_label": sample["human_label"],
                "total_tokens": tokens,
                "protocol": result.get("protocol", self.protocol_name.upper()),
            }

            # Protocol-specific fields
            if self.protocol_name == "more":
                sample_record["scores"] = result.get("scores")
            else:
                sample_record["rounds_completed"] = result.get("rounds_completed")
                sample_record["stop_reason"] = result.get("stop_reason")
                sample_record["scores_per_round"] = result.get("scores_per_round")

            if result.get("error"):
                sample_record["error"] = result["error"]

            if result.get("verdict"):
                v = result["verdict"]
                sample_record["jury_winner"] = v.get("winner")
                sample_record["majority_reached"] = v.get("majority_reached")

            # --- Swapped order (for position bias) ---
            if measure_bias:
                protocol_swap = self._create_protocol()
                swap_result = protocol_swap.run(
                    question=sample["question"],
                    answer1=sample["answer2"],   # swapped
                    answer2=sample["answer1"],   # swapped
                )
                swap_winner = swap_result["winner"]
                token_counts.append(swap_result["total_tokens"])

                # Remap: if swapped run says 1, that means original answer2
                # won, so in original terms the winner is 2 (and vice versa).
                if swap_winner == 1:
                    remapped = 2
                elif swap_winner == 2:
                    remapped = 1
                else:
                    remapped = 0

                swapped_predictions.append(remapped)
                sample_record["swap_prediction_raw"] = swap_winner
                sample_record["swap_prediction_remapped"] = remapped
                sample_record["swap_tokens"] = swap_result["total_tokens"]

            per_sample.append(sample_record)

        elapsed = time.time() - start_time

        # --- Compute aggregate metrics ---
        agreement = metrics.agreement_rate(predictions, human_labels)
        kappa = metrics.cohens_kappa(predictions, human_labels)
        cost = metrics.cost_summary(token_counts)

        bias = None
        if measure_bias and swapped_predictions:
            # Compare original predictions vs remapped swapped predictions
            # Position bias: does the system give the same positional winner
            # regardless of which content is in that position?
            bias = metrics.position_bias(predictions, swapped_predictions)

        report = {
            "results": per_sample,
            "agreement": agreement,
            "kappa": kappa,
            "bias": bias,
            "cost": cost,
            "metadata": {
                "dataset": dataset_name,
                "protocol": self.protocol_name.upper(),
                "model": self.config.model,
                "num_samples": total,
                "measure_bias": measure_bias,
                "elapsed_seconds": round(elapsed, 2),
                "num_advocates": self.config.num_advocates,
                "max_rounds": self.config.max_rounds,
                "token_budget": self.config.token_budget,
                "use_jury": self.config.use_jury,
            },
        }

        # Print summary
        metrics.print_report(agreement, bias, kappa, cost)
        logger.info("Batch evaluation completed in %.1fs.", elapsed)

        # Optionally save to disk
        if output_path:
            self._save_results(report, output_path)

        return report

    @staticmethod
    def _save_results(report, output_path):
        """Write full report to JSON file.

        Args:
            report: The report dict from run().
            output_path: Destination file path.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Make serialisable: convert tuples to lists
        def _convert(obj):
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_convert)

        logger.info("Results written to %s", output_path)
        print(f"\nFull results saved to: {output_path}")
