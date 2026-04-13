"""
Benchmark loaders for D3 evaluation datasets.

The D3 paper evaluates on three pairwise-comparison benchmarks
(Section 4.1, Table 1):

  MT-Bench   — 80 multi-turn questions, 2-model answer pairs, human prefs
               (Zheng et al., 2024)
  AlignBench — Chinese LLM evaluation, 683 samples, human annotations
               (Liu et al., 2023)
  AUTO-J     — 1380 samples, pairwise + single-answer evaluation
               (Li et al., 2024)

Each benchmark sample is normalised to a common schema:
    {
        "question_id": str | int,
        "question": str,
        "answer1": str,
        "answer2": str,
        "human_label": 1 | 2 | 0,   # 0 = tie / not provided
        "metadata": dict,            # dataset-specific extras
    }

Expected data formats:
  MT-Bench   — JSONL with fields: question_id, question, model_a, model_b,
               answer_a, answer_b, winner  (from chatbot-arena style export)
  AlignBench — JSON array with fields: id, question, answer1, answer2, label
  AUTO-J     — JSONL with fields: id, input, output_1, output_2, label

Place data files under  data/<dataset_name>/  or pass a custom path.

Reference: Section 4.1, Tables 1-3 of the D3 paper (arXiv:2410.04663)
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# Project root is one level up from evaluation/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# Default file names expected inside data/<dataset>/
_DEFAULT_FILES = {
    "mt-bench": "mt_bench_pairwise.jsonl",
    "alignbench": "alignbench_pairwise.json",
    "auto-j": "autoj_pairwise.jsonl",
}


class BenchmarkLoader:
    """Load and manage evaluation benchmark datasets.

    Usage:
        loader = BenchmarkLoader("mt-bench")
        samples = loader.load()            # uses default data/ path
        samples = loader.load("path.jsonl") # custom file

        for sample in loader:
            result = protocol.run(
                question=sample["question"],
                answer1=sample["answer1"],
                answer2=sample["answer2"],
            )
    """

    SUPPORTED_DATASETS = ["mt-bench", "alignbench", "auto-j"]

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name: One of "mt-bench", "alignbench", "auto-j".

        Raises:
            ValueError: If dataset_name is not supported.
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )
        self.dataset_name = dataset_name
        self.data = []

    def load(self, data_path=None):
        """Load the benchmark dataset from disk.

        Args:
            data_path: Path to the dataset file. If None, looks in
                       data/<dataset_name>/<default_file>.

        Returns:
            list[dict]: Normalised evaluation samples.

        Raises:
            FileNotFoundError: If the data file does not exist.
            ValueError: If the file contains no valid samples.
        """
        if data_path is None:
            data_path = os.path.join(
                _DATA_DIR, self.dataset_name,
                _DEFAULT_FILES[self.dataset_name],
            )

        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"Dataset file not found: {data_path}\n"
                f"Download the {self.dataset_name} dataset and place it at "
                f"that path, or pass a custom data_path= argument."
            )

        logger.info("Loading %s from %s", self.dataset_name, data_path)

        if self.dataset_name == "mt-bench":
            self.data = self._load_mt_bench(data_path)
        elif self.dataset_name == "alignbench":
            self.data = self._load_alignbench(data_path)
        elif self.dataset_name == "auto-j":
            self.data = self._load_auto_j(data_path)

        if not self.data:
            raise ValueError(
                f"No valid samples found in {data_path} for "
                f"dataset '{self.dataset_name}'."
            )

        logger.info("Loaded %d samples from %s.", len(self.data), self.dataset_name)
        return self.data

    # ------------------------------------------------------------------
    # Dataset-specific parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mt_bench(path):
        """Parse MT-Bench pairwise comparison JSONL.

        Expected fields per line:
            question_id, question (or prompt), model_a, model_b,
            answer_a (or response_a), answer_b (or response_b),
            winner (or human_label): "model_a" | "model_b" | "tie"
        """
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("MT-Bench: skipping malformed line %d", lineno)
                    continue

                question = obj.get("question") or obj.get("prompt", "")
                answer1 = obj.get("answer_a") or obj.get("response_a", "")
                answer2 = obj.get("answer_b") or obj.get("response_b", "")

                if not question or not answer1 or not answer2:
                    logger.warning(
                        "MT-Bench: skipping line %d (missing question/answers)",
                        lineno,
                    )
                    continue

                # Parse human label
                raw_winner = str(
                    obj.get("winner") or obj.get("human_label", "tie")
                ).lower().strip()

                if raw_winner in ("model_a", "1", "a"):
                    human_label = 1
                elif raw_winner in ("model_b", "2", "b"):
                    human_label = 2
                else:
                    human_label = 0  # tie or unknown

                samples.append({
                    "question_id": obj.get("question_id", lineno),
                    "question": question,
                    "answer1": answer1,
                    "answer2": answer2,
                    "human_label": human_label,
                    "metadata": {
                        "model_a": obj.get("model_a", ""),
                        "model_b": obj.get("model_b", ""),
                        "category": obj.get("category", ""),
                        "source": "mt-bench",
                    },
                })
        return samples

    @staticmethod
    def _load_alignbench(path):
        """Parse AlignBench evaluation data.

        AlignBench is natively a *single-answer* benchmark: each sample
        has a question and a reference answer, but no pairwise comparison.
        The D3 paper creates pairwise samples by generating answers from
        two different models and comparing them.

        This loader supports two formats:

        1. **Native AlignBench** (JSONL or JSON array from THUDM/AlignBench):
           Fields: question_id, question, reference, category
           -> Loaded with reference as answer1, answer2 left empty.
              The runner must generate answer2 from a model before evaluation,
              or the user can provide a pre-generated pairwise JSON.

        2. **Pre-processed pairwise** (JSON array):
           Fields: id, question, answer1/response_1, answer2/response_2, label
           -> Loaded directly as pairwise samples.
        """
        # Detect format: JSONL (one JSON object per line) vs JSON array
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1).strip()

        if first_char == "[" or first_char == "{":
            # JSON array or dict wrapper
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = raw.get("data") or raw.get("samples") or []
            objects = raw
        else:
            # JSONL
            objects = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        objects.append(json.loads(line))

        samples = []
        for idx, obj in enumerate(objects):
            question = obj.get("question") or obj.get("prompt", "")

            if not question:
                logger.warning("AlignBench: skipping sample %d (no question)", idx)
                continue

            # Try pairwise fields first
            answer1 = (
                obj.get("answer1") or obj.get("response_1")
                or obj.get("answer_a", "")
            )
            answer2 = (
                obj.get("answer2") or obj.get("response_2")
                or obj.get("answer_b", "")
            )

            # Fall back to native AlignBench format (single-answer)
            if not answer1:
                answer1 = obj.get("reference", "")

            # Parse label
            if answer1 and answer2:
                # Pairwise format — parse human label
                raw_label = obj.get("label", 0)
                if isinstance(raw_label, str):
                    raw_label = raw_label.strip()
                    if raw_label in ("1", "a", "A"):
                        human_label = 1
                    elif raw_label in ("2", "b", "B"):
                        human_label = 2
                    else:
                        human_label = 0
                else:
                    human_label = int(raw_label) if raw_label in (1, 2) else 0
            else:
                # Single-answer format — no pairwise label
                human_label = 0

            samples.append({
                "question_id": obj.get("question_id") or obj.get("id", idx),
                "question": question,
                "answer1": answer1,
                "answer2": answer2,  # may be "" for native format
                "human_label": human_label,
                "metadata": {
                    "category": obj.get("category", ""),
                    "subcategory": obj.get("subcategory", ""),
                    "source": "alignbench",
                    "requires_generation": answer2 == "",
                },
            })
        return samples

    @staticmethod
    def _load_auto_j(path):
        """Parse AUTO-J pairwise comparison JSONL.

        Expected fields per line:
            id, input (or question), output_1, output_2,
            label: 1 | 2 | "tie"
        """
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("AUTO-J: skipping malformed line %d", lineno)
                    continue

                question = obj.get("input") or obj.get("question", "")
                answer1 = obj.get("output_1") or obj.get("response_1", "")
                answer2 = obj.get("output_2") or obj.get("response_2", "")

                if not question or not answer1 or not answer2:
                    logger.warning(
                        "AUTO-J: skipping line %d (missing fields)", lineno,
                    )
                    continue

                raw_label = str(obj.get("label", "tie")).strip().lower()
                if raw_label in ("1", "a"):
                    human_label = 1
                elif raw_label in ("2", "b"):
                    human_label = 2
                else:
                    human_label = 0

                samples.append({
                    "question_id": obj.get("id", lineno),
                    "question": question,
                    "answer1": answer1,
                    "answer2": answer2,
                    "human_label": human_label,
                    "metadata": {
                        "category": obj.get("category", ""),
                        "scenario": obj.get("scenario", ""),
                        "source": "auto-j",
                    },
                })
        return samples

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_sample(self, index):
        """Return a single sample by index.

        Raises:
            IndexError: If data is empty or index is out of range.
        """
        if not self.data:
            raise IndexError("No data loaded. Call load() first.")
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return (
            f"BenchmarkLoader(dataset='{self.dataset_name}', "
            f"samples={len(self.data)})"
        )
