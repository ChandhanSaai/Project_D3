"""
SAMRE Protocol: Single-Advocate Multi-Round Evaluation.

Iteratively refines arguments under an explicit token budget
and convergence checks.
"""

from agents.advocate import Advocate
from agents.judge import Judge
from agents.jury import Jury
from utils.budget import TokenBudgetManager


class SAMREProtocol:
    """Single-Advocate Multi-Round Evaluation protocol."""

    def __init__(self, config):
        self.config = config
        self.pro_advocate = Advocate(role="pro", model=config.model, advocate_id=0)
        self.con_advocate = Advocate(role="con", model=config.model, advocate_id=1)
        self.judge = Judge(model=config.model)
        self.jury = Jury(model=config.model, jury_size=config.jury_size) if config.use_jury else None
        self.budget_manager = TokenBudgetManager(config.token_budget)

    def run(self, question=None, response=None):
        """Run the SAMRE evaluation protocol.

        Args:
            question: The prompt/question to evaluate.
            response: The LLM response to evaluate.

        Returns:
            dict with evaluation results.
        """
        scores = []
        all_arguments = []
        round_num = 0

        # Phase 1: Debate - Iterative argument refinement
        pro_arg = self.pro_advocate.generate_argument(question, response)
        con_arg = self.con_advocate.generate_argument(question, response)
        self.budget_manager.consume(pro_arg["token_count"] + con_arg["token_count"])

        for round_num in range(1, self.config.max_rounds + 1):
            all_arguments.append({"round": round_num, "pro": pro_arg, "con": con_arg})

            # Phase 2: Deliberate - Judge scores this round
            judgment = self.judge.evaluate(
                question, response, [pro_arg, con_arg]
            )
            self.budget_manager.consume(judgment["token_count"])
            scores.append(judgment["score"])

            # Check convergence and budget
            if self.judge.check_convergence(scores, self.config.convergence_threshold):
                break
            if self.budget_manager.is_exhausted():
                break

            # Refine arguments for next round
            pro_arg = self.pro_advocate.refine_argument(con_arg["argument"])
            con_arg = self.con_advocate.refine_argument(pro_arg["argument"])
            self.budget_manager.consume(pro_arg["token_count"] + con_arg["token_count"])

        # Phase 3: Decide
        final_score = scores[-1] if scores else 0

        if self.jury:
            final = self.jury.deliberate(
                question, response, all_arguments, final_score
            )
            self.budget_manager.consume(final["token_count"])
            final_score = final["final_score"]

        return {
            "score": final_score,
            "rounds": round_num,
            "scores_per_round": scores,
            "arguments": all_arguments,
            "total_tokens": self.budget_manager.tokens_used,
            "budget_remaining": self.budget_manager.remaining(),
            "protocol": "SAMRE",
        }
