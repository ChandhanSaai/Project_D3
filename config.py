"""
Configuration for the D3 framework.

Reference: https://arxiv.org/abs/2410.04663
GitHub: https://github.com/abirharrasse/D3-Judge
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Auto-load .env file from project root
load_dotenv()


@dataclass
class D3Config:
    """Configuration for D3 evaluation framework."""

    # Model settings
    model: str = "gpt-4-turbo"  # Backbone LLM for all agent roles
    api_key: str = ""  # Set via environment variable OPENAI_API_KEY

    # MORE protocol settings
    num_advocates: int = 3  # k advocates per answer (k=3 in paper experiments)

    # SAMRE protocol settings
    max_rounds: int = 5  # Max debate rounds (paper: 58% converge by round 2)
    token_budget: int = 4096  # Token budget for cost-aware budgeted stopping

    # Convergence settings (SAMRE budgeted stopping rule)
    convergence_threshold: float = 0.05  # Score gap stability threshold

    # Jury settings
    use_jury: bool = True
    jury_size: int = 5  # Paper uses 5 jurors with diverse personas
    jury_personas: tuple = (
        "a retired professor of ethics",
        "a young environmental activist",
        "a middle-aged business owner",
        "a social worker specializing in community development",
        "a technology entrepreneur with a background in AI",
    )

    # Scoring criteria (Judge evaluates on 1-20 scale per criterion)
    scoring_criteria: tuple = (
        "Relevance to the question",
        "Accuracy of information and use of credible sources",
        "Depth of analysis and completeness of argument",
        "Clarity of expression and logical flow",
        "Strength of reasoning and factual support",
        "Effectiveness in addressing opponent's points",
    )

    # Dataset
    dataset: str = "mt-bench"

    # Output
    output_dir: str = "results"
    verbose: bool = True

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
