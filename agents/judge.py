# """
# Judge Agent for D3 Framework.

# The Judge acts as a moderator and facilitator of the debate.  Its primary
# function is to provide structured, criterion-based feedback on the arguments
# presented by the advocates.  It scores each side's defense on a predefined
# rubric (Relevance, Accuracy, Reasoning, ...).

# Scoring serves two purposes:
#   1. Signal for iterative refinement in multi-round debates (SAMRE).
#   2. Tie-breaking mechanism in the final jury decision.

# Algorithm mapping:
#   - MORE  (Alg 1, step 11): Judge evaluates aggregated defenses once   → score()
#   - SAMRE (Alg 2, step 5):  s_r_1, s_r_2, F_r ← Evaluate(J, d1, d2)  → evaluate()
#     Each round returns scores AND feedback in a single call.

# Reference: Section 2.1, Appendix D.2, Appendix F of the D3 paper
#            (arXiv:2410.04663)
# """

# import logging
# import os
# import re

# import tiktoken
# from openai import OpenAI

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Scoring rubric (Appendix D.2)
# #   Six criteria, each scored 1-20.  Max total = 120 per answer.
# # ---------------------------------------------------------------------------
# SCORING_CRITERIA = (
#     "Relevance to the question",
#     "Accuracy of information and use of credible sources",
#     "Depth of analysis and completeness of argument",
#     "Clarity of expression and logical flow",
#     "Strength of reasoning and factual support",
#     "Effectiveness in addressing opponent's points",
# )

# # ---------------------------------------------------------------------------
# # Prompt templates (from Appendix F of the paper)
# #
# # User-supplied fields are wrapped in <<<...>>> delimiters to reduce the
# # prompt-injection surface.
# # ---------------------------------------------------------------------------

# # F.1.1 — MORE: Full criterion-based scoring (single round)
# # Also reusable as a standalone scorer in any context.
# MORE_JUDGE_PROMPT = """You're a critical, impartial judge in a high-stakes debate on the question below.
# Your goal is to provide detailed, constructive feedback that will push advocates to significantly improve their arguments.

# Question:
# <<<{question}>>>

# Answer 1:
# <<<{answer1}>>>

# Answer 2:
# <<<{answer2}>>>

# Current round: {current_round}
# Max rounds: {max_rounds}
# Previous scores: {previous_scores}

# Defense for 1st answer:
# <<<{defense1}>>>

# Defense for 2nd answer:
# <<<{defense2}>>>

# Analyze each argument meticulously. Be thorough and unbiased in your assessment of:
# 1. Relevance to the question
# 2. Accuracy of information and use of credible sources
# 3. Depth of analysis and completeness of argument
# 4. Clarity of expression and logical flow
# 5. Strength of reasoning and factual support
# 6. Effectiveness in addressing opponent's points

# For each criterion, provide a score on a scale of 1-20 and detailed justification.
# Scores should be given as [Answer1_score, Answer2_score] for each criterion.

# Your comprehensive feedback for each advocate (50 words each):
# Feedback for Advocate 1:
# Feedback for Advocate 2:

# After your full evaluation, your FINAL LINE MUST be exactly in this format:
# FINAL_TUPLE: (score1, score2)

# Example:
# FINAL_TUPLE: (95, 87)

# Do not put anything after that final line.
# Your detailed scores and final tally:"""

# # F.2.4 — SAMRE: Per-round evaluation (scores + feedback together)
# # This is the prompt used at EVERY round of the SAMRE debate.
# # It produces s_r_1, s_r_2, F_r in one call per Algorithm 2 step 5.
# SAMRE_EVALUATE_PROMPT = """You're a critical, impartial judge in a high-stakes debate on the question below.
# Your goal is to provide detailed, constructive feedback that will push advocates to significantly improve their arguments.

# Question:
# <<<{question}>>>

# Answer 1:
# <<<{answer1}>>>

# Answer 2:
# <<<{answer2}>>>

# Current round: {current_round}
# Total rounds: {total_rounds}
# Previous scores: {previous_scores}

# Defense for 1st answer:
# <<<{defense1}>>>

# Defense for 2nd answer:
# <<<{defense2}>>>

# Analyze each argument meticulously. Be thorough and unbiased in your assessment of:
# 1. Relevance to the question
# 2. Accuracy of information and use of credible sources
# 3. Depth of analysis and completeness of argument
# 4. Clarity of expression and logical flow
# 5. Strength of reasoning and factual support
# 6. Effectiveness in addressing opponent's points

# For each criterion, provide a score on a scale of 1-20 and detailed justification.
# Scores should be given as [Answer1_score, Answer2_score] for each criterion.

# Your comprehensive feedback for each advocate (50 words each):
# Feedback for Advocate 1:
# Feedback for Advocate 2:

# Sum up the scores and return the final score tuple (score1, score2). Example: (95, 87)
# Your detailed scores and final tally:"""

# # F.1.3 — Summarizer prompt
# # Condenses transcript content while preserving score tuples.
# SUMMARIZER_PROMPT = """Summarize the following content in 50 words or less, if there are any scores tuples, return them, it's important! Start summarization directly, no introductory sentences like here's your summary. In your summarization, only focus on the last scores, no partial ones. This is important: return the tuple of scores.

# Content:
# <<<{content}>>>"""

# # Regex to extract score tuples like (95, 87) from LLM output
# _SCORE_TUPLE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


# class Judge:
#     """Judge agent that evaluates advocate arguments and decides scores.

#     The Judge is designed to be impartial. It never argues for a side;
#     it only scores and provides feedback on the advocacy presented.

#     Public methods map to algorithms in the paper:
#         score()       → MORE  (Algorithm 1, single-round evaluation)
#         evaluate()    → SAMRE (Algorithm 2 step 5, per-round scores + feedback)
#         summarize()   → Transcript condensation before jury deliberation
#     """

#     def __init__(self, model, api_key=None):
#         """
#         Args:
#             model: LLM model identifier (e.g., 'gpt-4-turbo').
#             api_key: OpenAI API key.  Falls back to OPENAI_API_KEY env var.

#         Raises:
#             ValueError: If the API key is missing.
#         """
#         self.model = model

#         # Validate API key early
#         resolved_key = api_key or os.getenv("OPENAI_API_KEY")
#         if not resolved_key:
#             raise ValueError(
#                 "OpenAI API key not found. "
#                 "Set the OPENAI_API_KEY environment variable or pass api_key."
#             )
#         self.client = OpenAI(api_key=resolved_key)

#         # Token estimator
#         try:
#             self.encoder = tiktoken.encoding_for_model(model)
#         except KeyError:
#             self.encoder = tiktoken.get_encoding("cl100k_base")

#         # Running history for this evaluation session
#         self.score_history = []

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def estimate_prompt_tokens(self, prompt):
#         """Estimate token count *before* an API call (for budget checks).

#         Args:
#             prompt: The prompt string.

#         Returns:
#             int: Estimated token count.
#         """
#         return len(self.encoder.encode(prompt))

#     def _call_llm(self, system_message, prompt, max_completion_tokens=800):
#         """Make an LLM call with error handling.

#         Args:
#             system_message: The system-role instruction.
#             prompt: The user-role prompt.
#             max_completion_tokens: Maximum response tokens.

#         Returns:
#             dict with 'text', 'token_count', and 'success'.
#         """
#         try:
            
#             response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": prompt},
#             ],
#             temperature=0.3,
#             max_completion_tokens=max_completion_tokens,
#         )

#             text = response.choices[0].message.content.strip()
#             token_count = response.usage.total_tokens
#             return {"text": text, "token_count": token_count, "success": True}
            
            
            
#             # response = self.client.chat.completions.create(
#             #     model=self.model,
#             #     messages=[
#             #         {"role": "system", "content": system_message},
#             #         {"role": "user", "content": prompt},
#             #     ],
#             #     temperature=0.3,  # Lower temp for more consistent scoring
#             #     max_completion_tokens=max_completion_tokens,
#             # )

#             # text = response.choices[0].message.content.strip()
#             # token_count = response.usage.total_tokens
#             # return {"text": text, "token_count": token_count, "success": True}

#         except Exception as e:
#             logger.error("Judge LLM call failed: %s", e)
#             return {
#                 "text": f"[Judge LLM call failed: {e}]",
#                 "token_count": 0,
#                 "success": False,
#             }

#     @staticmethod
#     def parse_score_tuple(text):
#         """Extract the *last* (score1, score2) tuple from LLM output.

#         The paper's prompts ask the judge to end with a tuple like (95, 87).
#         We grab the last match because the final tally appears at the end.

#         Args:
#             text: Raw LLM output text.

#         Returns:
#             tuple[int, int] | None: Parsed scores or None if not found.
#         """
#         matches = _SCORE_TUPLE_RE.findall(text)
#         if not matches:
#             return None
#         last = matches[-1]
#         return (int(last[0]), int(last[1]))

#     @staticmethod
#     def _extract_feedback(raw_text):
#         """Best-effort extraction of per-advocate feedback blocks.

#         Looks for "Feedback for Advocate 1:" and "Feedback for Advocate 2:"
#         sections in the judge's raw output.

#         Args:
#             raw_text: Full judge response.

#         Returns:
#             tuple[str, str]: (feedback_a1, feedback_a2).  Empty strings on miss.
#         """
#         fb1 = ""
#         fb2 = ""

#         m1 = re.search(
#             r"Feedback for Advocate 1:\s*(.+?)(?=Feedback for Advocate 2:|Sum up|Your detailed|$)",
#             raw_text,
#             re.DOTALL | re.IGNORECASE,
#         )
#         if m1:
#             fb1 = m1.group(1).strip()

#         m2 = re.search(
#             r"Feedback for Advocate 2:\s*(.+?)(?=Sum up|Your detailed|$)",
#             raw_text,
#             re.DOTALL | re.IGNORECASE,
#         )
#         if m2:
#             fb2 = m2.group(1).strip()

#         return fb1, fb2

#     def _build_result(self, result):
#         """Shared post-processing for score() and evaluate().

#         Parses scores + feedback from the raw LLM output, appends to
#         score_history, and returns a uniform dict.

#         Args:
#             result: dict from _call_llm().

#         Returns:
#             dict with scores, raw, feedback_a1, feedback_a2,
#                   token_count, success.
#         """
#         if not result["success"]:
            
#             # parse_ok = scores is not None

#         # return {
#         #     "scores": scores,
#         #     "raw": raw,
#         #     "feedback_a1": feedback_a1,
#         #     "feedback_a2": feedback_a2,
#         #     "token_count": result["token_count"],
#         #     "success": parse_ok,
#         #     "parse_ok": parse_ok,
#         # }
            
            
#             return {
#                 "scores": None,
#                 "raw": result["text"],
#                 "feedback_a1": "",
#                 "feedback_a2": "",
#                 "token_count": result["token_count"],
#                 "success": False,
#             }

#         raw = result["text"]
#         scores = self.parse_score_tuple(raw)

#         # Store in history for convergence checks
#         if scores is not None:
#             self.score_history.append(scores)
#         else:
#             logger.warning(
#                 "Could not parse score tuple from judge output. "
#                 "Raw output (first 200 chars): %s",
#                 raw[:200],
#             )

#         feedback_a1, feedback_a2 = self._extract_feedback(raw)

#         return {
#             "scores": scores,
#             "raw": raw,
#             "feedback_a1": feedback_a1,
#             "feedback_a2": feedback_a2,
#             "token_count": result["token_count"],
#             "success": True,
#         }

#     # ------------------------------------------------------------------
#     # Public API — MORE
#     # ------------------------------------------------------------------

#     def score(
#         self,
#         question,
#         answer1,
#         answer2,
#         defense1,
#         defense2,
#         current_round=1,
#         max_rounds=1,
#         previous_scores="None yet",
#     ):
#         """Full criterion-based scoring of two defenses (Appendix F.1.1).

#         Used in MORE (Algorithm 1): single-round evaluation of the
#         aggregated defenses.  One call, one final score tuple.

#         Args:
#             question: The original question/prompt.
#             answer1: Candidate answer 1.
#             answer2: Candidate answer 2.
#             defense1: Aggregated defense for answer 1.
#             defense2: Aggregated defense for answer 2.
#             current_round: Current debate round number.
#             max_rounds: Maximum allowed rounds.
#             previous_scores: String summary of prior round scores.

#         Returns:
#             dict with keys:
#                 scores      — (score1, score2) or None on parse failure
#                 raw         — full LLM text (for transcript / interpretability)
#                 feedback_a1 — extracted feedback for advocate 1
#                 feedback_a2 — extracted feedback for advocate 2
#                 token_count — total tokens consumed
#                 success     — bool
#         """
#         prompt = MORE_JUDGE_PROMPT.format(
#             question=question,
#             answer1=answer1,
#             answer2=answer2,
#             current_round=current_round,
#             max_rounds=max_rounds,
#             previous_scores=previous_scores,
#             defense1=defense1,
#             defense2=defense2,
#         )

#         result = self._call_llm(
#             system_message=(
#                 "You are a critical, impartial judge. Evaluate debate "
#                 "arguments strictly on the stated criteria. Always end "
#                 "your response with the final score tuple (score1, score2)."
#             ),
#             prompt=prompt,
#             max_completion_tokens=800,
#         )

#         return self._build_result(result)

#     # ------------------------------------------------------------------
#     # Public API — SAMRE
#     # ------------------------------------------------------------------

#     def evaluate(
#         self,
#         question,
#         answer1,
#         answer2,
#         defense1,
#         defense2,
#         current_round=1,
#         total_rounds=5,
#         previous_scores="None yet",
#     ):
#         """Per-round evaluation for SAMRE: scores + feedback (Appendix F.2.4).

#         Maps directly to Algorithm 2, step 5:
#             s_r_1, s_r_2, F_r ← Evaluate(J, d_r_1, d_r_2)

#         Every SAMRE round calls this method ONCE and receives back:
#         - Scores (s_r_1, s_r_2) for convergence checks & budget tracking
#         - Feedback (F_r) to pass to advocates for argument refinement

#         Args:
#             question: The original question/prompt.
#             answer1: Candidate answer 1.
#             answer2: Candidate answer 2.
#             defense1: Current round defense for answer 1.
#             defense2: Current round defense for answer 2.
#             current_round: Current round number.
#             total_rounds: Max rounds configured.
#             previous_scores: Prior round scores as string.

#         Returns:
#             dict with keys:
#                 scores      — (score1, score2) or None on parse failure
#                 raw         — full LLM text (for transcript)
#                 feedback_a1 — constructive feedback for advocate 1
#                 feedback_a2 — constructive feedback for advocate 2
#                 token_count — total tokens consumed
#                 success     — bool
#         """
#         prompt = SAMRE_EVALUATE_PROMPT.format(
#             question=question,
#             answer1=answer1,
#             answer2=answer2,
#             current_round=current_round,
#             total_rounds=total_rounds,
#             previous_scores=previous_scores,
#             defense1=defense1,
#             defense2=defense2,
#         )

#         result = self._call_llm(
#             system_message=(
#                 "You are a critical, impartial judge in an iterative debate. "
#                 "Score each side on the stated criteria AND provide "
#                 "constructive feedback to help advocates refine their "
#                 "arguments for the next round. Always end your response "
#                 "with the final score tuple (score1, score2)."
#             ),
#             prompt=prompt,
#             max_completion_tokens=800,
#         )

#         return self._build_result(result)

#     # ------------------------------------------------------------------
#     # Public API — Summarizer
#     # ------------------------------------------------------------------

#     def summarize(self, content):
#         """Summarize debate content while preserving score tuples (Appendix F.1.3).

#         Used to condense transcripts before passing to the jury.

#         Args:
#             content: Raw transcript or argument text to summarize.

#         Returns:
#             dict with 'summary', 'token_count', and 'success'.
#         """
#         prompt = SUMMARIZER_PROMPT.format(content=content)

#         result = self._call_llm(
#             system_message="You are a concise summarizer. Preserve all score tuples exactly.",
#             prompt=prompt,
#             max_completion_tokens=200,
#         )

#         return {
#             "summary": result["text"],
#             "token_count": result["token_count"],
#             "success": result["success"],
#         }

#     # ------------------------------------------------------------------
#     # Convergence logic (Section 2.2 — Budgeted Stopping Rule)
#     # ------------------------------------------------------------------

#     def check_convergence(self, threshold):
#         """Check if the debate score gap has stabilised (SAMRE stopping rule).

#         The debate terminates when the score *difference* between consecutive
#         rounds is smaller than ``threshold``.  This implements the budgeted
#         stopping rule described in Section 2.2.

#         Args:
#             threshold: Maximum allowed change in score gap between rounds.

#         Returns:
#             bool: True if converged (safe to stop).
#         """
#         if len(self.score_history) < 2:
#             return False

#         prev_gap = abs(self.score_history[-2][0] - self.score_history[-2][1])
#         curr_gap = abs(self.score_history[-1][0] - self.score_history[-1][1])

#         return abs(curr_gap - prev_gap) < threshold

#     def get_cumulative_scores(self):
#         """Return cumulative (summed) scores across all rounds.

#         Used as the tie-breaker when the jury vote is split
#         (Section 2.3, Step 3).

#         Returns:
#             tuple[int, int]: (total_score1, total_score2) across all rounds.
#         """
#         if not self.score_history:
#             return (0, 0)

#         total1 = sum(s[0] for s in self.score_history)
#         total2 = sum(s[1] for s in self.score_history)
#         return (total1, total2)

#     def reset(self):
#         """Reset score history for a new evaluation."""
#         self.score_history = []













"""
Judge Agent for D3 Framework.

The Judge acts as a moderator and facilitator of the debate. Its primary
function is to provide structured, criterion-based feedback on the arguments
presented by the advocates. It scores each side's defense on a predefined
rubric (Relevance, Accuracy, Reasoning, ...).

Scoring serves two purposes:
  1. Signal for iterative refinement in multi-round debates (SAMRE).
  2. Tie-breaking mechanism in the final jury decision.

Algorithm mapping:
  - MORE  (Alg 1, step 11): Judge evaluates aggregated defenses once -> score()
  - SAMRE (Alg 2, step 5):  s_r_1, s_r_2, F_r <- Evaluate(J, d1, d2) -> evaluate()
    Each round returns scores AND feedback in a single call.

Reference: Section 2.1, Appendix D.2, Appendix F of the D3 paper
           (arXiv:2410.04663)
"""

import logging
import os
import re

import tiktoken
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring rubric (Appendix D.2)
#   Six criteria, each scored 1-20. Max total = 120 per answer.
# ---------------------------------------------------------------------------
SCORING_CRITERIA = (
    "Relevance to the question",
    "Accuracy of information and use of credible sources",
    "Depth of analysis and completeness of argument",
    "Clarity of expression and logical flow",
    "Strength of reasoning and factual support",
    "Effectiveness in addressing opponent's points",
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MORE_JUDGE_PROMPT = """You're a critical, impartial judge in a high-stakes debate on the question below.
Your goal is to provide detailed, constructive feedback that will push advocates to significantly improve their arguments.

Question:
<<<{question}>>>

Answer 1:
<<<{answer1}>>>

Answer 2:
<<<{answer2}>>>

Current round: {current_round}
Max rounds: {max_rounds}
Previous scores: {previous_scores}

Defense for 1st answer:
<<<{defense1}>>>

Defense for 2nd answer:
<<<{defense2}>>>

Analyze each argument meticulously. Be thorough and unbiased in your assessment of:
1. Relevance to the question
2. Accuracy of information and use of credible sources
3. Depth of analysis and completeness of argument
4. Clarity of expression and logical flow
5. Strength of reasoning and factual support
6. Effectiveness in addressing opponent's points

For each criterion, provide a score on a scale of 1-20 and detailed justification.
Scores should be given as [Answer1_score, Answer2_score] for each criterion.

Your comprehensive feedback for each advocate (50 words each):
Feedback for Advocate 1:
Feedback for Advocate 2:

After your full evaluation, your FINAL LINE MUST be exactly in this format:
FINAL_TUPLE: (score1, score2)

Example:
FINAL_TUPLE: (95, 87)

Do not put anything after that final line.
Your detailed scores and final tally:"""

SAMRE_EVALUATE_PROMPT = """You're a critical, impartial judge in a high-stakes debate on the question below.
Your goal is to provide detailed, constructive feedback that will push advocates to significantly improve their arguments.

Question:
<<<{question}>>>

Answer 1:
<<<{answer1}>>>

Answer 2:
<<<{answer2}>>>

Current round: {current_round}
Total rounds: {total_rounds}
Previous scores: {previous_scores}

Defense for 1st answer:
<<<{defense1}>>>

Defense for 2nd answer:
<<<{defense2}>>>

Analyze each argument meticulously. Be thorough and unbiased in your assessment of:
1. Relevance to the question
2. Accuracy of information and use of credible sources
3. Depth of analysis and completeness of argument
4. Clarity of expression and logical flow
5. Strength of reasoning and factual support
6. Effectiveness in addressing opponent's points

For each criterion, provide a score on a scale of 1-20 and detailed justification.
Scores should be given as [Answer1_score, Answer2_score] for each criterion.

Your comprehensive feedback for each advocate (50 words each):
Feedback for Advocate 1:
Feedback for Advocate 2:

After your full evaluation, your FINAL LINE MUST be exactly in this format:
FINAL_TUPLE: (score1, score2)

Example:
FINAL_TUPLE: (95, 87)

Do not put anything after that final line.
Your detailed scores and final tally:"""

SUMMARIZER_PROMPT = """Summarize the following content in 50 words or less. If there are any score tuples, preserve them exactly. Start summarization directly, with no introductory sentence. Focus on the latest scores only.

Content:
<<<{content}>>>"""

# Strict primary parser + fallback parser
_FINAL_SCORE_TUPLE_RE = re.compile(
    r"FINAL_TUPLE:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
    re.IGNORECASE,
)
_SCORE_TUPLE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


class Judge:
    """Judge agent that evaluates advocate arguments and decides scores."""

    def __init__(self, model, api_key=None):
        """
        Args:
            model: LLM model identifier.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

        Raises:
            ValueError: If the API key is missing.
        """
        self.model = model

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set the OPENAI_API_KEY environment variable or pass api_key."
            )
        self.client = OpenAI(api_key=resolved_key)

        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.score_history = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def estimate_prompt_tokens(self, prompt):
        """Estimate token count before an API call."""
        return len(self.encoder.encode(prompt))

    def _call_llm(self, system_message, prompt, max_completion_tokens=2000):
        """Make an LLM call with robust response handling.

        Returns:
            dict with 'text', 'token_count', and 'success'.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=max_completion_tokens,
            )

            message = response.choices[0].message
            content = getattr(message, "content", None)

            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", ""))
                    else:
                        parts.append(getattr(part, "text", ""))
                text = "".join(parts).strip()
            else:
                text = (content or "").strip()

            token_count = (
                getattr(response.usage, "total_tokens", 0)
                if getattr(response, "usage", None) else 0
            )

            if not text:
                logger.error(
                    "Judge returned empty content. finish_reason=%s token_count=%s",
                    getattr(response.choices[0], "finish_reason", None),
                    token_count,
                )
                return {"text": "", "token_count": token_count, "success": False}

            return {"text": text, "token_count": token_count, "success": True}

        except Exception as e:
            logger.error("Judge LLM call failed: %s", e)
            return {
                "text": f"[Judge LLM call failed: {e}]",
                "token_count": 0,
                "success": False,
            }

    @staticmethod
    def parse_score_tuple(text):
        """Extract final score tuple from judge output."""
        m = _FINAL_SCORE_TUPLE_RE.search(text)
        if m:
            return (int(m.group(1)), int(m.group(2)))

        # Fallback for older prompt variants
        matches = _SCORE_TUPLE_RE.findall(text)
        if not matches:
            return None
        last = matches[-1]
        return (int(last[0]), int(last[1]))

    @staticmethod
    def _extract_feedback(raw_text):
        """Best-effort extraction of per-advocate feedback blocks."""
        fb1 = ""
        fb2 = ""

        m1 = re.search(
        r"(?:#+\s*)?Feedback for Advocate 1(?:\s*\(.*?\))?\s*:?\s*(.+?)(?=(?:#+\s*)?Feedback for Advocate 2|FINAL_TUPLE:|$)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
        )
        if m1:
            fb1 = m1.group(1).strip()

        m2 = re.search(
        r"(?:#+\s*)?Feedback for Advocate 2(?:\s*\(.*?\))?\s*:?\s*(.+?)(?=FINAL_TUPLE:|$)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
        )
        if m2:
            fb2 = m2.group(1).strip()

        return fb1, fb2

    def _build_result(self, result):
        """Shared post-processing for score() and evaluate()."""
        if not result["success"]:
            return {
                "scores": None,
                "raw": result["text"],
                "feedback_a1": "",
                "feedback_a2": "",
                "token_count": result["token_count"],
                "success": False,
                "parse_ok": False,
            }

        raw = result["text"]
        scores = self.parse_score_tuple(raw)

        if scores is not None:
            self.score_history.append(scores)
        else:
            logger.warning(
                "Could not parse score tuple from judge output. "
                "Raw output (first 200 chars): %s",
                raw[:200],
            )

        feedback_a1, feedback_a2 = self._extract_feedback(raw)
        parse_ok = scores is not None

        return {
            "scores": scores,
            "raw": raw,
            "feedback_a1": feedback_a1,
            "feedback_a2": feedback_a2,
            "token_count": result["token_count"],
            "success": parse_ok,
            "parse_ok": parse_ok,
        }

    # ------------------------------------------------------------------
    # Public API — MORE
    # ------------------------------------------------------------------

    def score(
        self,
        question,
        answer1,
        answer2,
        defense1,
        defense2,
        current_round=1,
        max_rounds=1,
        previous_scores="None yet",
    ):
        """Full criterion-based scoring of two defenses for MORE."""
        prompt = MORE_JUDGE_PROMPT.format(
            question=question,
            answer1=answer1,
            answer2=answer2,
            current_round=current_round,
            max_rounds=max_rounds,
            previous_scores=previous_scores,
            defense1=defense1,
            defense2=defense2,
        )

        result = self._call_llm(
            system_message=(
                "You are a critical, impartial judge. Evaluate debate "
                "arguments strictly on the stated criteria. Always end "
                "your response with a final line exactly formatted as "
                "FINAL_TUPLE: (score1, score2)."
            ),
            prompt=prompt,
            max_completion_tokens=2000,
        )

        return self._build_result(result)

    # ------------------------------------------------------------------
    # Public API — SAMRE
    # ------------------------------------------------------------------

    def evaluate(
        self,
        question,
        answer1,
        answer2,
        defense1,
        defense2,
        current_round=1,
        total_rounds=5,
        previous_scores="None yet",
    ):
        """Per-round evaluation for SAMRE: scores + feedback."""
        prompt = SAMRE_EVALUATE_PROMPT.format(
            question=question,
            answer1=answer1,
            answer2=answer2,
            current_round=current_round,
            total_rounds=total_rounds,
            previous_scores=previous_scores,
            defense1=defense1,
            defense2=defense2,
        )

        result = self._call_llm(
            system_message=(
                "You are a critical, impartial judge in an iterative debate. "
                "Score each side on the stated criteria AND provide "
                "constructive feedback to help advocates refine their "
                "arguments for the next round. Always end your response "
                "with a final line exactly formatted as "
                "FINAL_TUPLE: (score1, score2)."
            ),
            prompt=prompt,
            max_completion_tokens=2000,
        )

        return self._build_result(result)

    # ------------------------------------------------------------------
    # Public API — Summarizer
    # ------------------------------------------------------------------

    def summarize(self, content):
        """Summarize debate content while preserving score tuples."""
        prompt = SUMMARIZER_PROMPT.format(content=content)

        result = self._call_llm(
            system_message="You are a concise summarizer. Preserve all score tuples exactly.",
            prompt=prompt,
            max_completion_tokens=2000,
        )

        return {
            "summary": result["text"],
            "token_count": result["token_count"],
            "success": result["success"],
        }

    # ------------------------------------------------------------------
    # Convergence logic
    # ------------------------------------------------------------------

    def check_convergence(self, threshold):
        """Check if the debate score gap has stabilised."""
        if len(self.score_history) < 2:
            return False

        prev_gap = abs(self.score_history[-2][0] - self.score_history[-2][1])
        curr_gap = abs(self.score_history[-1][0] - self.score_history[-1][1])

        return abs(curr_gap - prev_gap) < threshold

    def get_cumulative_scores(self):
        """Return cumulative (summed) scores across all rounds."""
        if not self.score_history:
            return (0, 0)

        total1 = sum(s[0] for s in self.score_history)
        total2 = sum(s[1] for s in self.score_history)
        return (total1, total2)

    def reset(self):
        """Reset score history for a new evaluation."""
        self.score_history = []