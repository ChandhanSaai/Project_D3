"""
MORE Protocol: Multi-Advocate One-Round Evaluation.

Optimized for breadth and efficiency.  For each candidate answer, k
advocates generate arguments in parallel.  These are aggregated into a
single comprehensive defense per side.  The Judge evaluates the two
consolidated defenses in a single round.  A diverse jury panel then
deliberates on the anonymized transcript and renders a final verdict.

Algorithm 1 (paper):
  1:  Init advocates A1 = {A_11,...,A_1k}, A2 = {A_21,...,A_2k}
  2:  Init defenses  D1 <- {}, D2 <- {}
  3-8:  for i = 1 to k:  (parallel)
          d_1i <- GenerateArgument(A_1i, Answer1)
          D1 <- D1 U {d_1i}
          d_2i <- GenerateArgument(A_2i, Answer2)
          D2 <- D2 U {d_2i}
  9:  D_1,agg <- AggregateDefenses(D1)
  10: D_2,agg <- AggregateDefenses(D2)
  11: Compile transcript T with aggregated defenses + judge scores
  12-16: Jury deliberation
  17: winner <- AggregateVotes(V)  # tie-break with Judge's score

Reference: Section 2.2, Algorithm 1, Appendix F.1 of the D3 paper
           (arXiv:2410.04663)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.advocate import Advocate
from agents.judge import Judge
from agents.jury import Jury
from utils.budget import TokenBudgetManager

logger = logging.getLogger(__name__)


class MOREProtocol:
    """Multi-Advocate One-Round Evaluation protocol."""

    def __init__(self, config):
        """
        Args:
            config: D3Config instance.
        """
        self.config = config
        self.api_key = config.api_key

        # --- Step 1: Init k advocates per answer ---
        # A1 = {A_11, ..., A_1k} for Answer 1
        self.advocates_a1 = [
            Advocate(role="pro", model=config.model,
                     advocate_id=i, api_key=config.api_key)
            for i in range(config.num_advocates)
        ]
        # A2 = {A_21, ..., A_2k} for Answer 2
        self.advocates_a2 = [
            Advocate(role="con", model=config.model,
                     advocate_id=config.num_advocates + i,
                     api_key=config.api_key)
            for i in range(config.num_advocates)
        ]

        self.judge = Judge(model=config.model, api_key=config.api_key)

        self.jury = None
        if config.use_jury:
            self.jury = Jury(
                model=config.model,
                personas=config.jury_personas,
                api_key=config.api_key,
            )

        self.budget = TokenBudgetManager(config.token_budget)

    def run(self, question, answer1, answer2):
        """Run the full MORE evaluation protocol (Algorithm 1).

        Args:
            question: The original question/prompt.
            answer1: Candidate answer 1.
            answer2: Candidate answer 2.

        Returns:
            dict with:
                winner          — 1 (Answer 1), 2 (Answer 2), or 0 (tie)
                scores          — (score1, score2) from Judge
                verdict         — full jury verdict dict (if jury enabled)
                transcript      — compiled debate transcript
                defenses_a1     — list of individual defenses for answer 1
                defenses_a2     — list of individual defenses for answer 2
                aggregated_a1   — aggregated defense for answer 1
                aggregated_a2   — aggregated defense for answer 2
                judge_result    — full Judge scoring result
                total_tokens    — total tokens consumed
                protocol        — "MORE"
        """
        if self.config.verbose:
            logger.info(
                "Running MORE protocol: k=%d advocates per answer",
                self.config.num_advocates,
            )

        # ---------------------------------------------------------------
        # Steps 3-8: Generate k parallel arguments per answer
        # Paper specifies parallel generation (Algorithm 1, lines 3-8)
        # ---------------------------------------------------------------
        defenses_a1 = [None] * self.config.num_advocates
        defenses_a2 = [None] * self.config.num_advocates
        failures = []

        def _generate_a1(i):
            return ("a1", i, self.advocates_a1[i].generate_argument(
                question=question, answer=answer1, opponent_answer=answer2,
            ))

        def _generate_a2(i):
            return ("a2", i, self.advocates_a2[i].generate_argument(
                question=question, answer=answer2, opponent_answer=answer1,
            ))

        with ThreadPoolExecutor(
            max_workers=self.config.num_advocates * 2
        ) as executor:
            futures = []
            for i in range(self.config.num_advocates):
                futures.append(executor.submit(_generate_a1, i))
                futures.append(executor.submit(_generate_a2, i))

            for future in as_completed(futures):
                side, idx, result = future.result()
                self.budget.consume(result["token_count"])
                if not result.get("success", True):
                    failures.append(f"Advocate {side}[{idx}]")
                    logger.warning(
                        "Advocate %s[%d] failed: %s",
                        side, idx, result.get("argument", "no response"),
                    )
                if side == "a1":
                    defenses_a1[idx] = result
                else:
                    defenses_a2[idx] = result

        # Filter out failed advocate results (keep only successful ones)
        valid_a1 = [d for d in defenses_a1 if d and d.get("success", True)]
        valid_a2 = [d for d in defenses_a2 if d and d.get("success", True)]

        if not valid_a1 or not valid_a2:
            logger.error(
                "Too many advocate failures (a1=%d/%d, a2=%d/%d). Aborting.",
                len(valid_a1), self.config.num_advocates,
                len(valid_a2), self.config.num_advocates,
            )
            return {
                "winner": 0,
                "scores": None,
                "verdict": None,
                "transcript": "",
                "defenses_a1": defenses_a1,
                "defenses_a2": defenses_a2,
                "aggregated_a1": None,
                "aggregated_a2": None,
                "judge_result": None,
                "total_tokens": self.budget.tokens_used,
                "budget_remaining": self.budget.remaining(),
                "protocol": "MORE",
                "error": f"Advocate generation failed: {failures}",
            }

        if failures:
            logger.warning(
                "%d advocate(s) failed; continuing with %d+%d valid defenses.",
                len(failures), len(valid_a1), len(valid_a2),
            )

        # ---------------------------------------------------------------
        # Steps 9-10: AggregateDefenses
        # Use the first advocate of each side to aggregate (instance method)
        # ---------------------------------------------------------------
        agg_a1 = self.advocates_a1[0].aggregate_defenses(
            question=question,
            answer=answer1,
            opponent_answer=answer2,
            defenses=[d["argument"] for d in valid_a1],
        )
        self.budget.consume(agg_a1["token_count"])

        agg_a2 = self.advocates_a2[0].aggregate_defenses(
            question=question,
            answer=answer2,
            opponent_answer=answer1,
            defenses=[d["argument"] for d in valid_a2],
        )
        self.budget.consume(agg_a2["token_count"])

        # Check aggregation success
        if not agg_a1.get("success", True) or not agg_a2.get("success", True):
            logger.error("Defense aggregation failed. Aborting.")
            return {
                "winner": 0,
                "scores": None,
                "verdict": None,
                "transcript": "",
                "defenses_a1": defenses_a1,
                "defenses_a2": defenses_a2,
                "aggregated_a1": agg_a1.get("aggregated_defense"),
                "aggregated_a2": agg_a2.get("aggregated_defense"),
                "judge_result": None,
                "total_tokens": self.budget.tokens_used,
                "budget_remaining": self.budget.remaining(),
                "protocol": "MORE",
                "error": "Defense aggregation failed",
            }

        # ---------------------------------------------------------------
        # Judge scores the aggregated defenses (single round)
        # (Implicit between steps 10 and 11 — see Section 2.2)
        # ---------------------------------------------------------------
        judge_result = self.judge.score(
            question=question,
            answer1=answer1,
            answer2=answer2,
            defense1=agg_a1["aggregated_defense"],
            defense2=agg_a2["aggregated_defense"],
            current_round=1,
            max_rounds=1,
        )
        self.budget.consume(judge_result["token_count"])

        # Check judge scoring success
        if not judge_result.get("success", True) or judge_result.get("scores") is None:
            logger.error("Judge scoring failed. Aborting.")
            return {
                "winner": 0,
                "scores": None,
                "verdict": None,
                "transcript": "",
                "defenses_a1": defenses_a1,
                "defenses_a2": defenses_a2,
                "aggregated_a1": agg_a1["aggregated_defense"],
                "aggregated_a2": agg_a2["aggregated_defense"],
                "judge_result": judge_result,
                "total_tokens": self.budget.tokens_used,
                "budget_remaining": self.budget.remaining(),
                "protocol": "MORE",
                "error": "Judge scoring failed",
            }

        scores = judge_result["scores"]  # (score1, score2) or None

        # ---------------------------------------------------------------
        # Step 11: Compile transcript T
        # ---------------------------------------------------------------
        arguments = [{
            "round": 1,
            "defense1": agg_a1["aggregated_defense"],
            "defense2": agg_a2["aggregated_defense"],
        }]

        judge_scores = [scores]
        judge_fb_a1 = [judge_result.get("feedback_a1", "")]
        judge_fb_a2 = [judge_result.get("feedback_a2", "")]

        transcript = Jury.compile_transcript(
            question=question,
            answer1=answer1,
            answer2=answer2,
            arguments=arguments,
            judge_scores=judge_scores,
            judge_feedback_a1=judge_fb_a1,
            judge_feedback_a2=judge_fb_a2,
        )

        # ---------------------------------------------------------------
        # Steps 12-17: Jury deliberation + verdict aggregation
        # ---------------------------------------------------------------
        verdict = None
        if self.jury:
            jury_result = self.jury.deliberate(
                question=question,
                answer1=answer1,
                answer2=answer2,
                transcript=transcript,
                judge_cumulative_scores=self.judge.get_cumulative_scores(),
            )
            self.budget.consume(jury_result["total_tokens"])
            verdict = jury_result["verdict"]
            winner = verdict["winner"]
        else:
            # No jury: determine winner from judge scores alone
            if scores and scores[0] != scores[1]:
                winner = 1 if scores[0] > scores[1] else 2
            else:
                winner = 0

        return {
            "winner": winner,
            "scores": scores,
            "verdict": verdict,
            "transcript": transcript,
            "defenses_a1": defenses_a1,
            "defenses_a2": defenses_a2,
            "aggregated_a1": agg_a1["aggregated_defense"],
            "aggregated_a2": agg_a2["aggregated_defense"],
            "judge_result": judge_result,
            "total_tokens": self.budget.tokens_used,
            "budget_remaining": self.budget.remaining(),
            "protocol": "MORE",
        }

    def reset(self):
        """Reset all agents for a new evaluation."""
        for a in self.advocates_a1 + self.advocates_a2:
            a.reset()
        self.judge.reset()
        self.budget = TokenBudgetManager(self.config.token_budget)



# Example usage

# if __name__ == "__main__":
#     import dotenv
#     from config import D3Config

#     dotenv.load_dotenv()

#     config = D3Config(
#         model="gpt-5.4-nano",
#         api_key=None,
#         num_advocates=3,
#         use_jury=True,
#         jury_personas=None,
#         token_budget=20000,
#         verbose=True,
#     )

#     protocol = MOREProtocol(config)

#     question = "Which is better for hydration: water or soda?"
#     answer1 = "Water is better for hydration because it replenishes fluids without added sugar."
#     answer2 = "Soda is better because it contains sugar and energy."

#     result = protocol.run(question, answer1, answer2)
#     print(result)