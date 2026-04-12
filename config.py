"""Configuration for the D3 framework."""

from dataclasses import dataclass, field


@dataclass
class D3Config:
    """Configuration for D3 evaluation framework."""

    # Model settings
    model: str = "gpt-4"
    api_key: str = ""  # Set via environment variable OPENAI_API_KEY

    # Protocol settings
    num_advocates: int = 3  # k advocates for MORE protocol
    max_rounds: int = 5  # Max debate rounds for SAMRE protocol
    token_budget: int = 4096  # Token budget for cost-aware stopping

    # Agent settings
    use_jury: bool = False
    jury_size: int = 3

    # Convergence settings
    convergence_threshold: float = 0.05  # Score gap threshold for early stopping

    # Dataset
    dataset: str = "mt-bench"

    # Output
    output_dir: str = "results"
    verbose: bool = True
