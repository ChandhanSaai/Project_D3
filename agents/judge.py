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