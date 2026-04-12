"""
Benchmark loaders for D3 evaluation datasets.

Supports: MT-Bench, AlignBench, AUTO-J
"""


class BenchmarkLoader:
    """Load and manage evaluation benchmark datasets."""

    SUPPORTED_DATASETS = ["mt-bench", "alignbench", "auto-j"]

    def __init__(self, dataset_name):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )
        self.dataset_name = dataset_name
        self.data = []

    def load(self, data_path=None):
        """Load the benchmark dataset.

        Args:
            data_path: Path to the dataset file. If None, uses default path.

        Returns:
            List of evaluation samples.
        """
        # TODO: Implement dataset loading for each benchmark
        raise NotImplementedError(f"Loader for {self.dataset_name} not yet implemented")

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
