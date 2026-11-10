"""Exception hierarchy and structured logging for AI services.

Call configure_logging() explicitly before using the example. generate()
simulates a timeout without contacting a provider. Raw exception messages are
logged here for demonstration; sanitize them before using with real secrets.
"""

import time
import uuid
from functools import wraps
from typing import Any

import structlog


def configure_logging() -> None:
    """Configure console logging explicitly; use JSONRenderer in production."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )


log = structlog.get_logger()


class AIServiceError(Exception):
    """Base exception for the AI service."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class LLMProviderError(AIServiceError):
    """The LLM provider returned an error."""


class RateLimitError(LLMProviderError):
    """API rate limit exceeded — should retry with backoff."""

    def __init__(self, provider: str, retry_after: float | None = None):
        super().__init__(f"Rate limit exceeded for {provider}")
        self.retry_after = retry_after


class TokenLimitError(AIServiceError):
    """Input exceeds the model's context window."""

    def __init__(self, tokens: int, limit: int):
        super().__init__(f"Input has {tokens} tokens, limit is {limit}")
        self.tokens = tokens
        self.limit = limit


class VectorStoreError(AIServiceError):
    """Vector database operation failed."""


def log_llm_call(func):
    """Log async LLM calls with latency and top-level total_tokens if present."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())[:8]
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            model = kwargs.get("model", "unknown")
            log.info("llm_call_start", model=model)
            start = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                elapsed_ms = round((time.perf_counter() - start) * 1000)
                log.info(
                    "llm_call_success",
                    model=model,
                    latency_ms=elapsed_ms,
                    tokens=getattr(result, "total_tokens", None),
                )
                return result
            except RateLimitError:
                log.warning("llm_rate_limit", model=model)
                raise
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - start) * 1000)
                log.error(
                    "llm_call_failed",
                    model=model,
                    latency_ms=elapsed_ms,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                raise LLMProviderError(str(exc)) from exc

    return wrapper


@log_llm_call
async def generate(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Simulate a provider timeout without making a network request."""
    raise TimeoutError("Connection timed out")


if __name__ == "__main__":
    configure_logging()
