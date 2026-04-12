"""
MORE Protocol: Multi-Advocate One-Round Evaluation.

Elicits k parallel defenses per answer to amplify signal
via diverse advocacy in a single round.
"""

from agents.advocate import Advocate
from agents.judge import Judge
from agents.jury import Jury
from utils.scoring import aggregate_scores


class MOREProtocol:
    """Multi-Advocate One-Round Evaluation protocol."""

    def __init__(self, config):
        self.config = config
        self.advocates = [
            Advocate(role="pro", model=config.model, advocate_id=i)
            for i in range(config.num_advocates)
        ]
        self.judge = Judge(model=config.model)
        self.jury = Jury(model=config.model, jury_size=config.jury_size) if config.use_jury else None

    def run(self, question=None, response=None):
        """Run the MORE evaluation protocol.

        Args:
            question: The prompt/question to evaluate.
            response: The LLM response to evaluate.

        Returns:
            dict with evaluation results.
        """
        # Phase 1: Debate - k advocates generate parallel arguments
        arguments = []
        total_tokens = 0
        for advocate in self.advocates:
            result = advocate.generate_argument(question, response)
            arguments.append(result)
            total_tokens += result["token_count"]

        # Phase 2: Deliberate - Judge evaluates all arguments
        judgment = self.judge.evaluate(question, response, arguments)
        total_tokens += judgment["token_count"]

        # Phase 3: Decide - Optional jury deliberation
        if self.jury:
            final = self.jury.deliberate(
                question, response, arguments, judgment["score"]
            )
            total_tokens += final["token_count"]
            score = final["final_score"]
        else:
            score = judgment["score"]

        return {
            "score": score,
            "arguments": arguments,
            "judgment": judgment,
            "total_tokens": total_tokens,
            "protocol": "MORE",
        }
