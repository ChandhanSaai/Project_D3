"""
Jury Agent for D3 Framework.

The final decision rests with a panel of LLM agents assigned diverse,
predefined personas.  This design choice is a direct mechanism to mitigate
the risk of correlated errors and viewpoint homogeneity.

Deliberation process (Section 2.3):
  1. Transcript Compilation — complete anonymized transcript is compiled
     (question, answers, all arguments, all judge feedback/scores).
  2. Jury Deliberation — the full transcript is presented to each juror
     independently; each provides a final score and written rationale.
  3. Verdict Aggregation — majority vote of jurors.  Ties broken by the
     Judge's cumulative score from the debate phase.

Algorithm mapping (steps 12-17 in both Alg 1 and Alg 2):
    V <- {}
    for each Juror C_i in panel:
        v_i <- Vote(C_i, T)
        V <- V U {v_i}
    winner <- AggregateVotes(V)   # tie-break with Judge's score

Notation (Appendix D.1):
    f_C_i(C_i, M_r): voting decision by juror C_i on final memory M_r.
    M_r: aggregated memory including arguments, scores, and feedback.

Reference: Sections 2.1, 2.3, 5.1, 5.8, Appendix D.1, D.3, G of the
           D3 paper (arXiv:2410.04663)
"""

import logging
import os
import re

import tiktoken
from openai import OpenAI

logger = logging.getLogger(__name__)

# Juror score range (prompt asks 1-100)
_JUROR_SCORE_MIN = 1
_JUROR_SCORE_MAX = 100
_MAX_VOTE_RETRIES = 2

# ---------------------------------------------------------------------------
# Default juror personas (Appendix D.3)
#
# Selected to provide complementary professional perspectives and value
# lenses.  Each persona is anchored in domain expertise rather than
# demographic attributes, reducing stereotype risks.  All jurors share the
# same backbone LLM; personas only influence instructional framing.
#
# Value coverage (Section G.2):
#   - Ethics & human values          (ethics professor)
#   - Social & environmental impact  (environmental activist, social worker)
#   - Business & practical trade-offs (business owner)
#   - Technology & innovation         (tech entrepreneur)
# ---------------------------------------------------------------------------
DEFAULT_PERSONAS = (
    "a retired professor of ethics",
    "a young environmental activist",
    "a middle-aged business owner",
    "a social worker specializing in community development",
    "a technology entrepreneur with a background in AI",
)

# ---------------------------------------------------------------------------
# Juror vote prompt
#
# The paper describes the process (Section 2.3) but does not include a
# verbatim juror prompt in Appendix F.  This template is faithful to:
#   - Independent evaluation of the anonymized transcript (2.3-step 2)
#   - Producing a final score for each answer AND a written rationale (2.3)
#   - Persona-guided reasoning style (Section 2.1, 5.1, G.2)
#   - Evidence citation from the transcript (Section 5.8)
# ---------------------------------------------------------------------------
JUROR_VOTE_PROMPT = """You are {persona}. You have been selected as a juror to evaluate a debate between two answers to a question.

Below is the complete, anonymized debate transcript. Your task is to independently evaluate both answers based on the arguments and evidence presented in the debate. You must be fair and thorough.

Debate transcript:
<<<{transcript}>>>

Instructions:
1. Carefully review the full debate transcript above.
2. Evaluate each answer based on the strength of the arguments and evidence presented by their advocates.
3. Reference specific points from the debate in your rationale (e.g., "Advocate A argued that...", "In round 2, the rebuttal addressed...").
4. Provide a score for each answer on a scale of 1-100.
5. State your verdict: which answer is better, or if it is a tie.

Your response MUST follow this exact format:
Score for Answer 1: [number]
Score for Answer 2: [number]
Verdict: [Answer 1 / Answer 2 / Tie]
Rationale: [Your detailed reasoning in 50-100 words, citing debate evidence]"""


class Jury:
    """A panel of persona-diverse jurors who independently evaluate debate transcripts.

    Each juror is an LLM instance guided by a distinct persona prompt.
    The jury produces a majority-vote verdict with interpretable rationales.
    """

    def __init__(self, model, personas=None, api_key=None):
        """
        Args:
            model: LLM model identifier (e.g., 'gpt-5.4-nano').
            personas: Tuple/list of persona descriptions.  Defaults to
                      the 5 curated personas from Appendix D.3.
            api_key: OpenAI API key.  Falls back to OPENAI_API_KEY env var.

        Raises:
            ValueError: If API key is missing or fewer than 2 personas given.
        """
        self.model = model
        self.personas = tuple(personas) if personas else DEFAULT_PERSONAS

        if len(self.personas) < 2:
            raise ValueError(
                "Jury requires at least 2 personas for meaningful deliberation, "
                f"got {len(self.personas)}."
            )

        # Validate API key early
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set the OPENAI_API_KEY environment variable or pass api_key."
            )
        self.client = OpenAI(api_key=resolved_key)

        # Token estimator
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def size(self):
        """Number of jurors on the panel."""
        return len(self.personas)

    def estimate_prompt_tokens(self, prompt):
        """Estimate token count before an API call (for budget checks)."""
        return len(self.encoder.encode(prompt))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system_message, prompt, max_tokens=500):
        """Make an LLM call with error handling.

        Args:
            system_message: System-role instruction.
            prompt: User-role prompt.
            max_tokens: Max response tokens.

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
                temperature=0.5,
                max_tokens=max_tokens,
            )

            text = response.choices[0].message.content.strip()
            token_count = response.usage.total_tokens
            return {"text": text, "token_count": token_count, "success": True}

        except Exception as e:
            logger.error("Juror LLM call failed: %s", e)
            return {
                "text": f"[Juror LLM call failed: {e}]",
                "token_count": 0,
                "success": False,
            }

    @staticmethod
    def _parse_juror_vote(text):
        """Parse a juror's structured response into score/verdict/rationale.

        Validates scores are within the 1-100 range.  If an explicit
        Verdict line is missing, the verdict is inferred from scores.

        Args:
            text: Raw juror response.

        Returns:
            dict with 'score1', 'score2', 'verdict', 'rationale', 'parse_ok'.
        """
        result = {
            "score1": None,
            "score2": None,
            "verdict": None,
            "rationale": "",
            "parse_ok": False,
        }

        # --- Extract and validate scores ---
        s1 = re.search(r"Score for Answer 1:\s*(\d+)", text, re.IGNORECASE)
        s2 = re.search(r"Score for Answer 2:\s*(\d+)", text, re.IGNORECASE)

        if s1:
            val = int(s1.group(1))
            if _JUROR_SCORE_MIN <= val <= _JUROR_SCORE_MAX:
                result["score1"] = val
            else:
                logger.warning(
                    "Juror score1=%d out of range [%d, %d], discarding.",
                    val, _JUROR_SCORE_MIN, _JUROR_SCORE_MAX,
                )

        if s2:
            val = int(s2.group(1))
            if _JUROR_SCORE_MIN <= val <= _JUROR_SCORE_MAX:
                result["score2"] = val
            else:
                logger.warning(
                    "Juror score2=%d out of range [%d, %d], discarding.",
                    val, _JUROR_SCORE_MIN, _JUROR_SCORE_MAX,
                )

        # --- Extract verdict ---
        v = re.search(r"Verdict:\s*(Answer\s*[12]|Tie)", text, re.IGNORECASE)
        if v:
            verdict_raw = v.group(1).strip().lower()
            if "1" in verdict_raw:
                result["verdict"] = 1
            elif "2" in verdict_raw:
                result["verdict"] = 2
            else:
                result["verdict"] = 0  # Tie

        # --- Extract rationale ---
        r_match = re.search(r"Rationale:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if r_match:
            result["rationale"] = r_match.group(1).strip()

        # --- Fallback: infer verdict from scores if explicit verdict missing ---
        if (
            result["verdict"] is None
            and result["score1"] is not None
            and result["score2"] is not None
        ):
            if result["score1"] > result["score2"]:
                result["verdict"] = 1
            elif result["score2"] > result["score1"]:
                result["verdict"] = 2
            else:
                result["verdict"] = 0

        result["parse_ok"] = (
            result["score1"] is not None
            and result["score2"] is not None
            and result["verdict"] is not None
        )

        return result

    # ------------------------------------------------------------------
    # Transcript compilation (Section 2.3, Step 1)
    #
    # "a complete, anonymized transcript is compiled, including the
    #  original question, candidate answers, all arguments, and all
    #  feedback and scores from the Judge."
    # ------------------------------------------------------------------

    @staticmethod
    def compile_transcript(
        question,
        answer1,
        answer2,
        arguments,
        judge_scores,
        judge_feedback_a1=None,
        judge_feedback_a2=None,
    ):
        """Compile a complete, anonymized debate transcript (Section 2.3 Step 1).

        Produces a self-contained transcript that includes the original
        question, candidate answers (anonymized), all advocate arguments,
        and all judge scores and per-advocate feedback.

        Advocate identities are replaced with generic labels
        (Advocate A / Advocate B) for anonymization.

        Args:
            question: The original question/prompt.
            answer1: Candidate answer 1.
            answer2: Candidate answer 2.
            arguments: List of round dicts, each containing defenses.
                Each round: {'round': int, 'defense1': str, 'defense2': str}
            judge_scores: List of (score1, score2) tuples per round.
            judge_feedback_a1: Optional list of feedback strings for
                advocate 1, one per round.  Matches Judge.evaluate()
                output 'feedback_a1'.
            judge_feedback_a2: Optional list of feedback strings for
                advocate 2, one per round.  Matches Judge.evaluate()
                output 'feedback_a2'.

        Returns:
            str: The formatted anonymized transcript.
        """
        lines = []

        # --- Include question and candidate answers (Section 2.3) ---
        lines.append("=== QUESTION ===")
        lines.append(question)
        lines.append("")
        lines.append("=== ANSWER 1 ===")
        lines.append(answer1)
        lines.append("")
        lines.append("=== ANSWER 2 ===")
        lines.append(answer2)
        lines.append("")
        lines.append("=== DEBATE ===")
        lines.append("")

        # --- Include all rounds of debate ---
        for i, round_data in enumerate(arguments):
            round_num = round_data.get("round", i + 1)
            lines.append(f"--- Round {round_num} ---")
            lines.append(
                f"Advocate A's defense: {round_data.get('defense1', 'N/A')}"
            )
            lines.append(
                f"Advocate B's defense: {round_data.get('defense2', 'N/A')}"
            )

            # Judge scores for this round
            if i < len(judge_scores) and judge_scores[i] is not None:
                s1, s2 = judge_scores[i]
                lines.append(
                    f"Judge scores: Answer 1 = {s1}, Answer 2 = {s2}"
                )

            # Per-advocate feedback (matches Judge.evaluate() output)
            if judge_feedback_a1 and i < len(judge_feedback_a1):
                lines.append(
                    f"Judge feedback for Advocate A: {judge_feedback_a1[i]}"
                )
            if judge_feedback_a2 and i < len(judge_feedback_a2):
                lines.append(
                    f"Judge feedback for Advocate B: {judge_feedback_a2[i]}"
                )

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vote(self, persona, question, answer1, answer2, transcript):
        """A single juror casts their vote (Algorithm 1/2, step 14).

        Corresponds to: v_i <- Vote(C_i, T)

        Retries up to _MAX_VOTE_RETRIES times on parse failure, since a
        single unparseable response can flip a 5-juror majority.

        Args:
            persona: This juror's persona description.
            question: The original question.
            answer1: Candidate answer 1.
            answer2: Candidate answer 2.
            transcript: The compiled anonymized transcript.

        Returns:
            dict with 'score1', 'score2', 'verdict', 'rationale',
                  'persona', 'token_count', 'success', 'parse_ok'.
        """
        # The transcript is now self-contained (includes question + answers),
        # so we don't duplicate them in the prompt.
        prompt = JUROR_VOTE_PROMPT.format(
            persona=persona,
            transcript=transcript,
        )

        system_message = (
            f"You are {persona}. You are serving as a juror in a "
            "structured debate evaluation. Evaluate independently and "
            "impartially based on the debate evidence presented. "
            "Follow the response format exactly."
        )

        total_tokens = 0

        for attempt in range(_MAX_VOTE_RETRIES + 1):
            result = self._call_llm(
                system_message=system_message,
                prompt=prompt,
                max_tokens=500,
            )
            total_tokens += result["token_count"]

            if not result["success"]:
                # LLM call itself failed (network, rate limit, etc.)
                if attempt < _MAX_VOTE_RETRIES:
                    logger.warning(
                        "Juror '%s' LLM call failed (attempt %d/%d), retrying.",
                        persona, attempt + 1, _MAX_VOTE_RETRIES + 1,
                    )
                    continue

                return {
                    "score1": None,
                    "score2": None,
                    "verdict": None,
                    "rationale": result["text"],
                    "raw": result["text"],
                    "persona": persona,
                    "token_count": total_tokens,
                    "success": False,
                    "parse_ok": False,
                }

            parsed = self._parse_juror_vote(result["text"])

            if parsed["parse_ok"] or attempt == _MAX_VOTE_RETRIES:
                # Either parsed OK, or we've exhausted retries
                parsed["persona"] = persona
                parsed["raw"] = result["text"]
                parsed["token_count"] = total_tokens
                parsed["success"] = True

                if not parsed["parse_ok"]:
                    logger.warning(
                        "Juror '%s' response could not be fully parsed "
                        "after %d attempt(s). Raw (first 200 chars): %s",
                        persona, attempt + 1, result["text"][:200],
                    )

                return parsed

            # Parse failed but we have retries left
            logger.warning(
                "Juror '%s' response could not be parsed (attempt %d/%d), "
                "retrying.",
                persona, attempt + 1, _MAX_VOTE_RETRIES + 1,
            )

        # Should not reach here, but safety fallback
        return {
            "score1": None,
            "score2": None,
            "verdict": None,
            "rationale": "",
            "raw": "",
            "persona": persona,
            "token_count": total_tokens,
            "success": False,
            "parse_ok": False,
        }

    def deliberate(self, question, answer1, answer2, transcript,
                   judge_cumulative_scores=None):
        """Full jury deliberation + verdict (Section 2.3, Steps 2-3).

        Implements Algorithm 1/2 steps 12-17 as a single operation:
          1. Each juror independently votes on the transcript.
          2. Votes are aggregated via majority rule.
          3. Ties are broken by the Judge's cumulative score.

        Args:
            question: The original question.
            answer1: Candidate answer 1.
            answer2: Candidate answer 2.
            transcript: Compiled anonymized debate transcript
                (from compile_transcript()).
            judge_cumulative_scores: Optional (total1, total2) from
                Judge.get_cumulative_scores() for tie-breaking.

        Returns:
            dict with:
                votes          — list of individual juror vote dicts
                verdict        — aggregated verdict dict (from aggregate_votes)
                total_tokens   — total tokens consumed across all jurors
                all_succeeded  — bool, whether all jurors parsed OK
        """
        votes = []
        total_tokens = 0

        for persona in self.personas:
            vote_result = self.vote(
                persona, question, answer1, answer2, transcript
            )
            votes.append(vote_result)
            total_tokens += vote_result["token_count"]

        all_succeeded = all(v["success"] and v["parse_ok"] for v in votes)

        if not all_succeeded:
            failed = [
                v["persona"]
                for v in votes
                if not v["success"] or not v["parse_ok"]
            ]
            logger.warning(
                "Some juror votes failed or could not be parsed: %s", failed
            )

        # Aggregate votes into final verdict (Algorithm step 17)
        verdict = self.aggregate_votes(votes, judge_cumulative_scores)

        return {
            "votes": votes,
            "verdict": verdict,
            "total_tokens": total_tokens,
            "all_succeeded": all_succeeded,
        }

    @staticmethod
    def aggregate_votes(votes, judge_cumulative_scores=None):
        """Aggregate jury votes into a final verdict (Section 2.3, Step 3).

        Corresponds to: winner <- AggregateVotes(V)

        Decision rule (Section 2.3 Step 3):
        1. Count votes for Answer 1, Answer 2, and Tie among valid votes.
        2. Strict majority required: an option must receive MORE than half
           of the valid votes (> total_valid / 2) to win outright.
        3. If no option has a strict majority, use the Judge's cumulative
           score as tie-breaker between Answer 1 and Answer 2.
        4. If tie-break is also tied or unavailable: returns 0 (tie).

        With 5 jurors, strict majority = 3+.  This matters when Tie votes
        split the count (e.g. 2 A, 1 B, 1 Tie = no majority despite A
        having the most votes).

        Args:
            votes: List of vote dicts from deliberate()['votes'].
            judge_cumulative_scores: Optional (total1, total2) from
                Judge.get_cumulative_scores() for tie-breaking.

        Returns:
            dict with:
                winner            — 1 (Answer 1), 2 (Answer 2), or 0 (tie)
                vote_counts       — {1: n, 2: n, 0: n}
                margin            — winning vote count minus second-highest
                tie_broken        — bool, whether judge tie-break was used
                majority_reached  — bool, whether strict majority was met
                rationales        — list of dicts with persona + rationale
        """
        valid_votes = [v for v in votes if v.get("verdict") is not None]

        counts = {0: 0, 1: 0, 2: 0}
        for v in valid_votes:
            counts[v["verdict"]] += 1

        rationales = [
            {
                "persona": v.get("persona", "unknown"),
                "rationale": v.get("rationale", ""),
            }
            for v in valid_votes
        ]

        total_valid = len(valid_votes)
        majority_threshold = total_valid / 2  # strict majority = MORE than half
        tie_broken = False
        majority_reached = False

        # Check each option for strict majority (> 50% of valid votes)
        majority_winner = None
        for option in (1, 2, 0):
            if counts[option] > majority_threshold:
                majority_winner = option
                break

        if majority_winner is not None:
            # Strict majority achieved
            winner = majority_winner
            majority_reached = True
            sorted_counts = sorted(counts.values(), reverse=True)
            margin = sorted_counts[0] - sorted_counts[1]
        else:
            # No strict majority — fall through to Judge tie-break
            # between Answer 1 and Answer 2 (Section 2.3 Step 3)
            margin = 0
            vote_1 = counts[1]
            vote_2 = counts[2]

            if judge_cumulative_scores:
                j1, j2 = judge_cumulative_scores
                if j1 > j2:
                    winner = 1
                    tie_broken = True
                elif j2 > j1:
                    winner = 2
                    tie_broken = True
                else:
                    winner = 0  # Judge also tied
            elif vote_1 != vote_2:
                # No judge scores available; use plurality between A1/A2
                # as a last resort (not in the paper, but better than
                # always returning tie when judge scores are absent).
                winner = 1 if vote_1 > vote_2 else 2
                margin = abs(vote_1 - vote_2)
            else:
                winner = 0  # True deadlock

        return {
            "winner": winner,
            "vote_counts": counts,
            "margin": margin,
            "tie_broken": tie_broken,
            "majority_reached": majority_reached,
            "rationales": rationales,
            "total_valid_votes": total_valid,
        }
