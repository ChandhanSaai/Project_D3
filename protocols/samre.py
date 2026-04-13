"""
SAMRE Protocol: Single-Advocate Multi-Round Evaluation with Budgeted Stopping.

Designed for depth and iterative refinement.  A single advocate for each
answer engages in a turn-based debate over multiple rounds.  In each round,
advocates use the Judge's feedback and their opponent's argument from the
previous round to refine their position.

Budgeted Stopping Rule (Section 2.2):
  The iterative debate terminates automatically if:
    (a) the debate has converged (score gap is stable), OR
    (b) a user-defined token budget is exceeded.

Algorithm 2 (paper):
  1:  Init advocates A1, A2, Judge J, Jurors {C1,...,Cm}, R_max, B
  2:  Init transcript T0 <- {}, scores S <- []
  3-11: for r = 1 to R_max:
    4:    d_r_1, d_r_2 <- GenerateArguments(A1, A2, T_{r-1})
    5:    s_r_1, s_r_2, F_r <- Evaluate(J, d_r_1, d_r_2)
    6:    S.append((s_r_1, s_r_2))
    7:    T_r <- T_{r-1} U {d_r_1, d_r_2, s_r_1, s_r_2, F_r}
    8-10: if CheckConvergence(S, e) or TokenCost(T_r) > B: break
  12-16: Jury deliberation on final transcript
  17: winner <- AggregateVotes(V)  # tie-break with Judge's final score

Key insight (Section 5.7):
  58% of debates converge by round 2; forced continuation beyond
  convergence changes verdicts in only 6% of cases.

Reference: Section 2.2, Algorithm 2, Appendix F.2 of the D3 paper
           (arXiv:2410.04663)
"""

import logging

from agents.advocate import Advocate
from agents.judge import Judge
from agents.jury import Jury
from utils.budget import TokenBudgetManager

logger = logging.getLogger(__name__)


class SAMREProtocol:
    """Single-Advocate Multi-Round Evaluation protocol with budgeted stopping."""

    def __init__(self, config):
        """
        Args:
            config: D3Config instance.
        """
        self.config = config

        # --- Step 1: Init one advocate per answer ---
        self.advocate_a1 = Advocate(
            role="pro", model=config.model,
            advocate_id=0, api_key=config.api_key,
        )
        self.advocate_a2 = Advocate(
            role="con", model=config.model,
            advocate_id=1, api_key=config.api_key,
        )

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
        """Run the full SAMRE evaluation protocol (Algorithm 2).

        Args:
            question: The original question/prompt.
            answer1: Candidate answer 1.
            answer2: Candidate answer 2.

        Returns:
            dict with:
                winner             — 1 (Answer 1), 2 (Answer 2), or 0 (tie)
                scores_per_round   — list of (s1, s2) per round
                verdict            — full jury verdict dict (if jury enabled)
                transcript         — compiled debate transcript
                rounds_completed   — number of debate rounds run
                stop_reason        — "convergence", "budget", or "max_rounds"
                judge_result_final — last judge evaluation result
                total_tokens       — total tokens consumed
                protocol           — "SAMRE"
        """
        if self.config.verbose:
            logger.info(
                "Running SAMRE protocol: max_rounds=%d, token_budget=%d",
                self.config.max_rounds, self.config.token_budget,
            )

        # --- Step 2: Init transcript, scores ---
        all_arguments = []     # round dicts for transcript compilation
        all_fb_a1 = []         # per-advocate feedback per round
        all_fb_a2 = []
        scores_per_round = []  # S in the algorithm
        stop_reason = "max_rounds"
        judge_result = None

        # Previous-round state for advocate refinement
        prev_defense_a1 = "None yet"
        prev_defense_a2 = "None yet"
        prev_feedback_for_a1 = "None yet"
        prev_feedback_for_a2 = "None yet"

        # ---------------------------------------------------------------
        # Steps 3-11: Iterative debate loop
        # ---------------------------------------------------------------
        for r in range(1, self.config.max_rounds + 1):

            # --- Step 4: Advocates generate/refine arguments ---
            # d_r_1, d_r_2 <- GenerateArguments(A1, A2, T_{r-1})
            # In SAMRE, advocates use feedback + opponent's last argument
            d1 = self.advocate_a1.defend(
                question=question,
                answer=answer1,
                opponent_answer=answer2,
                feedback=prev_feedback_for_a1,
                opponent_argument=prev_defense_a2,
                team_arguments=prev_defense_a1,
            )
            self.budget.consume(d1["token_count"])

            d2 = self.advocate_a2.defend(
                question=question,
                answer=answer2,
                opponent_answer=answer1,
                feedback=prev_feedback_for_a2,
                opponent_argument=prev_defense_a1,
                team_arguments=prev_defense_a2,
            )
            self.budget.consume(d2["token_count"])

            defense_a1 = d1["argument"]
            defense_a2 = d2["argument"]

            # --- Step 5: Judge scores + feedback ---
            # s_r_1, s_r_2, F_r <- Evaluate(J, d_r_1, d_r_2)
            prev_scores_str = (
                str(scores_per_round) if scores_per_round else "None yet"
            )

            judge_result = self.judge.evaluate(
                question=question,
                answer1=answer1,
                answer2=answer2,
                defense1=defense_a1,
                defense2=defense_a2,
                current_round=r,
                total_rounds=self.config.max_rounds,
                previous_scores=prev_scores_str,
            )
            self.budget.consume(judge_result["token_count"])

            # --- Step 5b: Handle judge failure ---
            scores = judge_result["scores"]  # (score1, score2) or None

            if not judge_result.get("success", True) or scores is None:
                logger.error(
                    "Judge scoring failed at round %d. Stopping debate.", r
                )
                stop_reason = "judge_failure"
                break

            # --- Step 6: S.append((s_r_1, s_r_2)) ---
            scores_per_round.append(scores)

            # --- Step 7: T_r <- T_{r-1} U {d_r_1, d_r_2, s_r_1, s_r_2, F_r} ---
            all_arguments.append({
                "round": r,
                "defense1": defense_a1,
                "defense2": defense_a2,
            })
            all_fb_a1.append(judge_result.get("feedback_a1", ""))
            all_fb_a2.append(judge_result.get("feedback_a2", ""))

            if self.config.verbose:
                logger.info(
                    "Round %d/%d: scores=%s, budget_used=%.1f%%",
                    r, self.config.max_rounds, scores,
                    self.budget.usage_fraction() * 100,
                )

            # --- Steps 8-10: Check stopping conditions ---
            # (a) Convergence check
            if self.judge.check_convergence(self.config.convergence_threshold):
                stop_reason = "convergence"
                if self.config.verbose:
                    logger.info("Debate converged at round %d.", r)
                break

            # (b) Budget check
            if self.budget.is_exhausted():
                stop_reason = "budget"
                if self.config.verbose:
                    logger.info("Token budget exhausted at round %d.", r)
                break

            # --- Prepare per-advocate feedback for next round ---
            # Each advocate receives their OWN feedback as "for your side"
            # and the opponent's feedback as "for opponent"
            prev_defense_a1 = defense_a1
            prev_defense_a2 = defense_a2
            fb1 = judge_result.get("feedback_a1", "")
            fb2 = judge_result.get("feedback_a2", "")
            prev_feedback_for_a1 = f"For your side: {fb1}\nFor opponent: {fb2}"
            prev_feedback_for_a2 = f"For your side: {fb2}\nFor opponent: {fb1}"

        rounds_completed = len(all_arguments)

        # ---------------------------------------------------------------
        # Early abort: no successful rounds means nothing to deliberate on
        # ---------------------------------------------------------------
        if rounds_completed == 0:
            logger.error(
                "No successful debate rounds completed (stop_reason=%s). "
                "Skipping transcript compilation and jury deliberation.",
                stop_reason,
            )
            return {
                "winner": 0,
                "scores_per_round": [],
                "verdict": None,
                "transcript": "",
                "rounds_completed": 0,
                "stop_reason": stop_reason,
                "judge_result_final": judge_result,
                "total_tokens": self.budget.tokens_used,
                "budget_remaining": self.budget.remaining(),
                "protocol": "SAMRE",
                "error": f"No completed rounds (stop_reason={stop_reason})",
            }

        # ---------------------------------------------------------------
        # Step 11 (post-loop): Compile final transcript
        # ---------------------------------------------------------------
        judge_scores_list = [s for s in scores_per_round]

        transcript = Jury.compile_transcript(
            question=question,
            answer1=answer1,
            answer2=answer2,
            arguments=all_arguments,
            judge_scores=judge_scores_list,
            judge_feedback_a1=all_fb_a1,
            judge_feedback_a2=all_fb_a2,
        )

        # ---------------------------------------------------------------
        # Steps 12-17: Jury deliberation on final transcript
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
            # No jury: use last round's judge scores
            last_scores = scores_per_round[-1] if scores_per_round else None
            if last_scores and last_scores[0] != last_scores[1]:
                winner = 1 if last_scores[0] > last_scores[1] else 2
            else:
                winner = 0

        return {
            "winner": winner,
            "scores_per_round": scores_per_round,
            "verdict": verdict,
            "transcript": transcript,
            "rounds_completed": rounds_completed,
            "stop_reason": stop_reason,
            "judge_result_final": judge_result,
            "total_tokens": self.budget.tokens_used,
            "budget_remaining": self.budget.remaining(),
            "protocol": "SAMRE",
        }

    def reset(self):
        """Reset all agents for a new evaluation."""
        self.advocate_a1.reset()
        self.advocate_a2.reset()
        self.judge.reset()
        self.budget = TokenBudgetManager(self.config.token_budget)
