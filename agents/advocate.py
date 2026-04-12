"""
Advocate Agent for D3 Framework.

Advocates are tasked with constructing the most compelling arguments
in favor of a specific candidate response. For a given question and
two answers, two sets of advocates work independently.

Their objective is not to be impartial but to be persuasive, focusing
on criteria such as factual accuracy, relevance, depth, and clarity.

Reference: Section 2.1 & Appendix F of the D3 paper (arXiv:2410.04663)
"""

import logging
import os

import tiktoken
from openai import OpenAI

logger = logging.getLogger(__name__)

VALID_ROLES = {"pro", "con"}

# ---------------------------------------------------------------------------
# Prompt templates (from Appendix F of the paper)
#
# All user-supplied fields are wrapped in <<<...>>> delimiters to reduce
# prompt-injection surface.  Not bulletproof, but better than inline prose
# stuffing (see review item #8).
# ---------------------------------------------------------------------------

# F.1.2 - MORE: Single-round advocate prompt
MORE_ADVOCATE_PROMPT = """You're a fierce advocate defending this answer to the question below. Your opponent defends a different answer. Convince the audience your argument is superior by addressing these criteria:
1. Relevance
2. Accuracy
3. Depth
4. Clarity
5. Logic and Factuality
Be assertive and don't hold back! Use rhetoric and persuasion to win over the audience but be respectful.

Question:
<<<{question}>>>

Defended answer:
<<<{answer}>>>

Opponent's answer:
<<<{opponent_answer}>>>

Latest feedback:
<<<{feedback}>>>

Opponent's last argument:
<<<{opponent_argument}>>>

Your fierce defense (100 words max):"""

# F.2.1 - SAMRE: Multi-round individual advocate prompt
SAMRE_DEFEND_PROMPT = """You're an eloquent advocate (Advocate {advocate_id}) in a group defending the answer below. Your opponent group defends a different answer. Collaborate with your fellow advocates to convince the audience your argument is better. Use the latest feedback, your opponent's last argument, and your team's previous arguments to improve your case.

Question:
<<<{question}>>>

Defended answer:
<<<{answer}>>>

Opponent's answer:
<<<{opponent_answer}>>>

Latest feedback:
<<<{feedback}>>>

Opponent's last argument:
<<<{opponent_argument}>>>

Your team's previous arguments:
<<<{team_arguments}>>>

Respond in under 80 words.
Your defense:"""

# F.2.2 - Used in BOTH protocols:
#   - MORE: consolidates k parallel defenses into one cohesive argument
#   - SAMRE: can aggregate group arguments when multiple advocates per side
AGGREGATE_DEFENSE_PROMPT = """You are an expert debate strategist. Your task is to aggregate and improve upon the following defenses for the answer below. Analyze each defense critically. Identify the strongest points, address any weaknesses, and combine the best arguments into a cohesive, powerful defense. Aim to create a defense that is stronger and more comprehensive than any individual argument.

Question:
<<<{question}>>>

Defended answer:
<<<{answer}>>>

Opponent's answer:
<<<{opponent_answer}>>>

Individual defenses:
<<<{defenses}>>>

Latest feedback from the judge:
<<<{feedback}>>>

Provide your aggregated and improved defense in under 150 words:"""


class Advocate:
    """An advocate agent that argues in favor of a specific LLM response.

    In D3's courtroom-inspired architecture, advocates are persuasive agents
    whose outputs are anonymized before being entered into the debate record.
    """

    def __init__(self, role, model, advocate_id=0, api_key=None):
        """
        Args:
            role: 'pro' or 'con' - indicates which answer this advocate defends.
            model: LLM model identifier (e.g., 'gpt-4-turbo').
            advocate_id: Unique identifier for this advocate.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

        Raises:
            ValueError: If role is not 'pro' or 'con', or API key is missing.
        """
        # --- Fix #7: Validate role early ---
        if role not in VALID_ROLES:
            raise ValueError(
                f"role must be one of {VALID_ROLES}, got '{role}'"
            )
        self.role = role
        self.model = model
        self.advocate_id = advocate_id
        self.argument_history = []

        # --- Fix #1: Validate API key early ---
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set the OPENAI_API_KEY environment variable or pass api_key."
            )
        self.client = OpenAI(api_key=resolved_key)

        # Token counter
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def estimate_prompt_tokens(self, prompt):
        """Estimate token count of a prompt *before* sending it to the API.

        Useful for:
        - Pre-flight budget checks in SAMRE's budgeted stopping rule.
        - Logging estimated vs actual usage for cost analysis.

        Args:
            prompt: The prompt string to estimate.

        Returns:
            int: Estimated token count.
        """
        return len(self.encoder.encode(prompt))

    def _call_llm(self, prompt):
        """Make an LLM API call and return the response text and token count.

        Args:
            prompt: The formatted prompt string.

        Returns:
            dict with 'text', 'token_count', and 'success'.
        """
        # --- Fix #2: Wrap API call in try/except ---
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a persuasive debate advocate. Your goal "
                            "is to construct the most compelling arguments in "
                            "favor of the answer you are defending."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                # removed max_tokens to allow flexible response lengths; can be re-added if strict limits are needed
                # max_tokens=300,
            )

            text = response.choices[0].message.content.strip()
            token_count = response.usage.total_tokens

            return {"text": text, "token_count": token_count, "success": True}

        except Exception as e:
            logger.error("Advocate %s LLM call failed: %s", self.advocate_id, e)
            return {"text": f"[LLM call failed: {e}]", "token_count": 0, "success": False}

    def generate_argument(self, question, answer, opponent_answer, feedback="None yet", opponent_argument="None yet"):
        """Generate a persuasive argument defending the assigned answer (MORE protocol).

        Uses the MORE advocate prompt template (Appendix F.1.2).

        Args:
            question: The original question/prompt.
            answer: The answer this advocate is defending.
            opponent_answer: The opposing answer.
            feedback: Judge's latest feedback (if any).
            opponent_argument: Opponent's last argument (if any).

        Returns:
            dict with 'argument', 'token_count', 'advocate_id', 'role', and 'success'.
        """
        prompt = MORE_ADVOCATE_PROMPT.format(
            answer=answer,
            question=question,
            opponent_answer=opponent_answer,
            feedback=feedback,
            opponent_argument=opponent_argument,
        )

        result = self._call_llm(prompt)

        argument_data = {
            "argument": result["text"],
            "token_count": result["token_count"],
            "advocate_id": self.advocate_id,
            "role": self.role,
            "success": result["success"],
        }

        self.argument_history.append(argument_data)
        return argument_data

    def defend(self, question, answer, opponent_answer, feedback="None yet", opponent_argument="None yet", team_arguments="None yet"):
        """Generate a defense in the SAMRE multi-round protocol.

        Uses the SAMRE defend prompt template (Appendix F.2.1).

        Args:
            question: The original question/prompt.
            answer: The answer this advocate is defending.
            opponent_answer: The opposing answer.
            feedback: Judge's latest feedback.
            opponent_argument: Opponent's latest argument.
            team_arguments: Previous arguments from this advocate's team.

        Returns:
            dict with 'argument', 'token_count', 'advocate_id', 'role', and 'success'.
        """
        prompt = SAMRE_DEFEND_PROMPT.format(
            advocate_id=self.advocate_id,
            answer=answer,
            question=question,
            opponent_answer=opponent_answer,
            feedback=feedback,
            opponent_argument=opponent_argument,
            team_arguments=team_arguments,
        )

        result = self._call_llm(prompt)

        argument_data = {
            "argument": result["text"],
            "token_count": result["token_count"],
            "advocate_id": self.advocate_id,
            "role": self.role,
            "success": result["success"],
        }

        self.argument_history.append(argument_data)
        return argument_data

    # --- Fix #4: Converted from @staticmethod to instance method ---
    def aggregate_defenses(self, question, answer, opponent_answer, defenses, feedback="None yet"):
        """Aggregate multiple advocate defenses into a single cohesive argument.

        Used in both protocols (Appendix F.2.2):
        - MORE: consolidates k parallel defenses per answer.
        - SAMRE: can aggregate group arguments when multiple advocates per side.

        Args:
            question: The original question.
            answer: The answer being defended.
            opponent_answer: The opposing answer.
            defenses: List of individual defense strings.
            feedback: Judge's latest feedback.

        Returns:
            dict with 'aggregated_defense', 'token_count', and 'success'.
        """
        defenses_text = "\n\n".join(
            f"Defense {i+1}: {d}" for i, d in enumerate(defenses)
        )

        prompt = AGGREGATE_DEFENSE_PROMPT.format(
            answer=answer,
            question=question,
            opponent_answer=opponent_answer,
            defenses=defenses_text,
            feedback=feedback,
        )

        # --- Fix #2: Error handling for aggregation call ---
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert debate strategist specializing "
                            "in synthesizing multiple arguments into a powerful, "
                            "cohesive defense."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                
                # removed max_tokens to allow flexible response lengths; can be re-added if strict limits are needed
                #max_tokens=400,
            )

            text = response.choices[0].message.content.strip()
            token_count = response.usage.total_tokens

            return {
                "aggregated_defense": text,
                "token_count": token_count,
                "success": True,
            }

        except Exception as e:
            logger.error("Defense aggregation failed: %s", e)
            return {
                "aggregated_defense": f"[Aggregation failed: {e}]",
                "token_count": 0,
                "success": False,
            }

    def reset(self):
        """Reset the advocate's argument history for a new evaluation."""
        self.argument_history = []

#--- Example usage (can be removed or commented out in production) ---

# import dotenv
# dotenv.load_dotenv()  # Load environment variables from .env file if present

# question = "Which is better for hydration: water or soda?"
# answer_a = "Water is better for hydration because it replenishes fluids without added sugar."
# answer_b = "Soda is better because it contains sugar and energy."

# adv = Advocate(role="pro", model="gpt-5.4-nano")  # or your model name

# result = adv.generate_argument(
#     question=question,
#     answer=answer_b,
#     opponent_answer=answer_a,
#     feedback="Focus on factual accuracy.",
#     opponent_argument="Soda provides energy, so it is more useful."
# )

# print("\n=== generate_argument ===")
# print(result)

# result2 = adv.defend(
#     question=question,
#     answer=answer_b,
#     opponent_answer=answer_a,
#     feedback="Be sharper.",
#     opponent_argument="Soda gives quick benefits.",
#     team_arguments="Water is healthier and has no added sugar."
# )

# print("\n=== defend ===")
# print(result2)

# agg = adv.aggregate_defenses(
#     question=question,
#     answer=answer_b,
#     opponent_answer=answer_a,
#     defenses=[
#         "Water directly restores fluid balance.",
#         "Soda contains sugar, which does not improve hydration quality.",
#         "Water is the medically standard hydration choice."
#     ],
#     feedback="Combine the strongest factual points."
# )

# print("\n=== aggregate_defenses ===")
# print(agg)