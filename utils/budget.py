"""
Token Budget Manager for D3 Framework.

Implements cost-aware budgeted stopping for the SAMRE protocol.
"""


class TokenBudgetManager:
    """Manages token budget for cost-aware evaluation."""

    def __init__(self, total_budget):
        """
        Args:
            total_budget: Maximum number of tokens allowed.
        """
        self.total_budget = total_budget
        self.tokens_used = 0

    def consume(self, tokens):
        """Record token usage.

        Args:
            tokens: Number of tokens consumed.
        """
        self.tokens_used += tokens

    def remaining(self):
        """Return remaining token budget."""
        return max(0, self.total_budget - self.tokens_used)

    def is_exhausted(self):
        """Check if the token budget has been exhausted."""
        return self.tokens_used >= self.total_budget

    def usage_fraction(self):
        """Return the fraction of budget used (0.0 to 1.0)."""
        if self.total_budget == 0:
            return 1.0
        return min(self.tokens_used / self.total_budget, 1.0)
