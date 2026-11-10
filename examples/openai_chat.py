"""Demonstrate async OpenAI Chat Completions with usage and conversation history.

Run: python3 examples/openai_chat.py
Requires the openai package and OPENAI_API_KEY in the environment. Running this
example makes real API requests and incurs API costs.
"""

import asyncio

from openai import AsyncOpenAI


async def chat(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> dict:
    """Complete chat with usage tracking."""
    async with AsyncOpenAI() as client:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    choice = response.choices[0]
    return {
        "content": choice.message.content,
        "finish_reason": choice.finish_reason,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


async def multi_turn_conversation():
    """Multi-turn conversation maintaining history."""
    messages = [
        {"role": "system", "content": "You are a Python tutor. Be concise."},
    ]

    turns = [
        "What is a decorator in Python?",
        "Show me an example with arguments.",
        "How does functools.wraps help?",
    ]

    async with AsyncOpenAI() as client:
        for user_msg in turns:
            messages.append({"role": "user", "content": user_msg})

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
            )

            assistant_msg = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_msg})
            print(f"User: {user_msg}")
            print(f"AI: {assistant_msg[:100]}...")
            print(f"  (tokens: {response.usage.total_tokens})\n")


if __name__ == "__main__":
    asyncio.run(multi_turn_conversation())
