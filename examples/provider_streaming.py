"""OpenAI and Anthropic streaming with text accumulation and timing metrics.

Run: python3 examples/provider_streaming.py
Requires openai, anthropic, OPENAI_API_KEY, and ANTHROPIC_API_KEY. Execution
makes real API requests and may incur costs. The tokens list stores received
text fragments, not individual tokenizer tokens; total_tokens uses API usage.
Model availability and SDK compatibility must be checked before running.
"""

import asyncio
import time
from dataclasses import dataclass, field

import anthropic
from openai import AsyncOpenAI


@dataclass
class StreamResult:
    content: str = ""
    tokens: list[str] = field(default_factory=list)
    time_to_first_token_ms: float = 0
    total_ms: float = 0
    total_tokens: int = 0


async def stream_openai(prompt: str, model: str = "gpt-4o-mini") -> StreamResult:
    """Stream from OpenAI with time to first text fragment and total usage."""
    result = StreamResult()
    async with AsyncOpenAI() as client:
        start = time.perf_counter()
        first_token = False

        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
        )

        async with stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    if not first_token:
                        result.time_to_first_token_ms = (time.perf_counter() - start) * 1000
                        first_token = True
                    result.tokens.append(token)
                    result.content += token
                    print(token, end="", flush=True)

                if chunk.usage:
                    result.total_tokens = chunk.usage.total_tokens

        result.total_ms = (time.perf_counter() - start) * 1000
    print()
    return result


async def stream_anthropic(prompt: str) -> StreamResult:
    """Stream from Anthropic with time to first text fragment and total usage."""
    result = StreamResult()
    async with anthropic.AsyncAnthropic() as client:
        start = time.perf_counter()
        first_token = False

        async with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                if not first_token:
                    result.time_to_first_token_ms = (time.perf_counter() - start) * 1000
                    first_token = True
                result.tokens.append(text)
                result.content += text
                print(text, end="", flush=True)

            final_message = await stream.get_final_message()
            result.total_tokens = (
                final_message.usage.input_tokens + final_message.usage.output_tokens
            )

        result.total_ms = (time.perf_counter() - start) * 1000
    print()
    return result


async def main():
    prompt = "Write a haiku about vector databases."

    print("=== OpenAI ===")
    r1 = await stream_openai(prompt)
    print(f"TTFT: {r1.time_to_first_token_ms:.0f}ms, Total: {r1.total_ms:.0f}ms\n")

    print("=== Anthropic ===")
    r2 = await stream_anthropic(prompt)
    print(f"TTFT: {r2.time_to_first_token_ms:.0f}ms, Total: {r2.total_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
