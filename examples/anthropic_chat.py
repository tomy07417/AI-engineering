"""Anthropic Claude messages, system prompts, and extended thinking example.

Run: python3 examples/anthropic_chat.py
Requires the anthropic SDK with extended thinking support and ANTHROPIC_API_KEY.
Running this example makes real API requests and may incur costs. Model access
and thinking support depend on the provider and installed SDK version.
"""

import asyncio

import anthropic


async def chat_claude(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> dict:
    """Chat with Claude, whose system prompt is a separate parameter."""
    async with anthropic.AsyncAnthropic() as client:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    return {
        "content": message.content[0].text,
        "stop_reason": message.stop_reason,
        "model": message.model,
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
    }


async def claude_with_thinking(prompt: str) -> dict:
    """Use extended thinking for complex reasoning tasks."""
    async with anthropic.AsyncAnthropic() as client:
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            messages=[{"role": "user", "content": prompt}],
        )

    result = {"thinking": None, "answer": None}
    for block in message.content:
        if block.type == "thinking":
            result["thinking"] = block.thinking
        elif block.type == "text":
            result["answer"] = block.text

    return result


async def main():
    result = await chat_claude(
        "Explain the difference between RAG and fine-tuning in 3 sentences.",
        system="You are an AI engineering instructor. Be precise and practical.",
    )
    print(f"Claude: {result['content']}")
    print(f"Tokens: {result['usage']}\n")

    thinking_result = await claude_with_thinking(
        "Design a RAG system for a legal document search engine. "
        "Consider chunking strategy, embedding model choice, and retrieval approach."
    )
    print(f"Thinking: {(thinking_result['thinking'] or '')[:200]}...")
    print(f"Answer: {(thinking_result['answer'] or '')[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
