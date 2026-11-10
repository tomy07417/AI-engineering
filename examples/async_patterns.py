"""Async AI examples: concurrent calls, bounded concurrency, and token streaming."""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class LLMResponse:
    provider: str
    content: str
    latency_ms: float


async def call_llm(provider: str, prompt: str, delay: float = 1.0) -> LLMResponse:
    """Simulate an async LLM API call without contacting a real provider."""
    start = time.perf_counter()
    await asyncio.sleep(delay)
    elapsed = (time.perf_counter() - start) * 1000
    return LLMResponse(
        provider=provider,
        content=f"Response from {provider}: {prompt[:50]}...",
        latency_ms=round(elapsed, 1),
    )


async def fan_out_query(prompt: str) -> list[LLMResponse]:
    """Query multiple simulated LLMs concurrently and return all responses."""
    async with asyncio.TaskGroup() as tg:
        openai_task = tg.create_task(call_llm("openai", prompt, delay=0.8))
        anthropic_task = tg.create_task(call_llm("anthropic", prompt, delay=1.2))
        gemini_task = tg.create_task(call_llm("gemini", prompt, delay=0.5))

    return [openai_task.result(), anthropic_task.result(), gemini_task.result()]


async def rate_limited_batch(
    prompts: list[str],
    max_concurrent: int = 5,
) -> list[LLMResponse]:
    """Bound concurrent calls; this does not enforce a requests-per-minute quota."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(prompt: str) -> LLMResponse:
        async with semaphore:
            return await call_llm("openai", prompt, delay=0.3)

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(process_one(p)) for p in prompts]

    return [t.result() for t in tasks]


async def stream_tokens(prompt: str) -> AsyncIterator[str]:
    """Simulate streaming by yielding words, not actual model tokens."""
    words = f"This is a response to: {prompt}".split()
    for word in words:
        await asyncio.sleep(0.05)
        yield word + " "


async def main() -> None:
    # Fan-out: query three simulated providers concurrently.
    responses = await fan_out_query("What is RAG?")
    for r in responses:
        print(f"  {r.provider}: {r.latency_ms}ms")

    # Batch: process 20 prompts with at most five concurrent calls.
    prompts = [f"Question {i}" for i in range(20)]
    batch_results = await rate_limited_batch(prompts, max_concurrent=5)
    print(f"Batch: {len(batch_results)} responses")

    print("Streaming: ", end="")
    async for token in stream_tokens("Hello"):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
