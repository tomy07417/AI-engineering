"""Educational LLM retries, sliding-window throttling, and provider fallback.

Run: python3 examples/resilient_llm_client.py
Requires openai, anthropic, OPENAI_API_KEY, and ANTHROPIC_API_KEY. Running this
example makes real API calls and may incur costs. Fallback sends messages to
another provider. This is not a production-ready spending or rate-limit guard:
the limiter is not concurrency-safe, all exceptions are retried, and SDK-level
retries may multiply attempts. max_retries here counts total attempts.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field

import anthropic
from openai import AsyncOpenAI


@dataclass
class LLMConfig:
    model: str
    provider: str
    max_retries: int = 3
    timeout_seconds: float = 30.0


@dataclass
class RateLimiter:
    """Illustrative sliding-window limiter for sequential acquire calls only."""

    max_requests: int = 50
    window_seconds: float = 60.0
    _timestamps: list[float] = field(default_factory=list)

    async def acquire(self):
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self.window_seconds]
        if len(self._timestamps) >= self.max_requests:
            sleep_time = self._timestamps[0] + self.window_seconds - now
            await asyncio.sleep(sleep_time)
        self._timestamps.append(time.monotonic())


class ResilientLLMClient:
    """Demonstrate retries, throttling, and fallback between two providers."""

    def __init__(self):
        self.openai = AsyncOpenAI()
        self.anthropic = anthropic.AsyncAnthropic()
        self.rate_limiter = RateLimiter(max_requests=50, window_seconds=60)
        self.fallback_chain: list[LLMConfig] = [
            LLMConfig(model="gpt-4o-mini", provider="openai"),
            LLMConfig(model="claude-sonnet-4-20250514", provider="anthropic"),
        ]

    async def close(self):
        """Attempt to close both SDK clients."""
        try:
            await self.openai.close()
        finally:
            await self.anthropic.close()

    async def _call_openai(self, messages: list[dict], config: LLMConfig) -> str:
        response = await asyncio.wait_for(
            self.openai.chat.completions.create(
                model=config.model,
                messages=messages,
            ),
            timeout=config.timeout_seconds,
        )
        return response.choices[0].message.content

    async def _call_anthropic(self, messages: list[dict], config: LLMConfig) -> str:
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        response = await asyncio.wait_for(
            self.anthropic.messages.create(
                model=config.model,
                max_tokens=1024,
                system=system,
                messages=filtered,
            ),
            timeout=config.timeout_seconds,
        )
        return response.content[0].text

    async def _call_with_retries(self, messages: list[dict], config: LLMConfig) -> str:
        """Call one provider with exponential backoff and jitter."""
        call_fn = self._call_openai if config.provider == "openai" else self._call_anthropic

        for attempt in range(config.max_retries):
            try:
                await self.rate_limiter.acquire()
                return await call_fn(messages, config)
            except Exception as exc:
                if attempt == config.max_retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Retry {attempt + 1}/{config.max_retries} after {wait:.1f}s: {exc}")
                await asyncio.sleep(wait)

        raise RuntimeError("Unreachable")

    async def chat(self, messages: list[dict]) -> dict:
        """Try each configured provider in order until one succeeds."""
        errors = []
        for config in self.fallback_chain:
            try:
                content = await self._call_with_retries(messages, config)
                return {"content": content, "provider": config.provider, "model": config.model}
            except Exception as exc:
                errors.append(f"{config.provider}: {exc}")
                continue

        raise RuntimeError(f"All providers failed: {errors}")


async def main():
    client = ResilientLLMClient()
    try:
        result = await client.chat([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is RAG?"},
        ])
        print(f"[{result['provider']}] {result['content']}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
