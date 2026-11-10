"""Token counting, illustrative cost estimation, and in-memory usage tracking.

Run: python3 examples/token_budget.py
Requires tiktoken; encoding files may be downloaded on first use. Prices are
user-supplied examples, not verified current rates. Token estimates do not
replace provider usage, especially for non-OpenAI models or message overhead.
UsageTracker records consumption before checking the budget; it is not a
pre-request spending guard and does not persist usage.
"""

from dataclasses import dataclass

import tiktoken


# Illustrative USD prices per million tokens.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate text tokens, falling back to o200k_base for unknown models."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """Estimate text message tokens; unlike count_tokens, unknown models raise."""
    enc = tiktoken.encoding_for_model(model)
    tokens = 3  # Approximate assistant reply overhead.
    for msg in messages:
        tokens += 4  # Approximate per-message overhead.
        tokens += len(enc.encode(msg.get("content", "")))
        tokens += len(enc.encode(msg.get("role", "")))
    return tokens


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini",
) -> float:
    """Estimate USD cost, using gpt-4o-mini prices for unknown models."""
    prices = PRICING.get(model, PRICING["gpt-4o-mini"])
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return round(input_cost + output_cost, 6)


@dataclass
class UsageTracker:
    """Track consumption in memory and report budget overruns after recording."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    budget_usd: float = 10.0

    def record(self, input_tokens: int, output_tokens: int, model: str):
        cost = estimate_cost(input_tokens, output_tokens, model)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.call_count += 1
        if self.total_cost_usd > self.budget_usd:
            raise RuntimeError(
                f"Budget exceeded: ${self.total_cost_usd:.4f} > ${self.budget_usd}"
            )

    def summary(self) -> str:
        return (
            f"Calls: {self.call_count} | "
            f"Tokens: {self.total_input_tokens:,} in + "
            f"{self.total_output_tokens:,} out | "
            f"Cost: ${self.total_cost_usd:.4f} / ${self.budget_usd:.2f}"
        )


if __name__ == "__main__":
    text = "Explain the difference between RAG and fine-tuning for LLMs."
    tokens = count_tokens(text, "gpt-4o-mini")
    cost = estimate_cost(tokens, 500, "gpt-4o-mini")
    print(f"Input tokens: {tokens}, estimated cost: ${cost:.6f}")

    tracker = UsageTracker(budget_usd=5.0)
    tracker.record(1000, 500, "gpt-4o-mini")
    tracker.record(2000, 1000, "gpt-4o")
    print(tracker.summary())
