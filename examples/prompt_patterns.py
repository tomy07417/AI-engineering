"""Prompt examples: step-by-step prompting, few-shot, and answer voting.

Run: python3 examples/prompt_patterns.py
Requires openai and OPENAI_API_KEY. Execution makes real API calls and may
incur costs. Self-consistency here uses exact-string plurality, not semantic
agreement or a guarantee of correctness; ties select the first seen answer.
Use a positive n. Sampling is concurrent without a rate or concurrency limit.
"""

import asyncio
from collections import Counter

from openai import AsyncOpenAI


COT_SYSTEM = """You are a precise analyst. For every question:
1. Break the problem into steps
2. Work through each step explicitly
3. State your final answer clearly after "ANSWER:"

Always show your reasoning before answering."""


async def chain_of_thought(question: str) -> str:
    async with AsyncOpenAI() as client:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": COT_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
    return response.choices[0].message.content


FEW_SHOT_MESSAGES = [
    {"role": "system", "content": "Classify the customer intent. Respond with ONLY the category."},
    {"role": "user", "content": "I can't log into my account"},
    {"role": "assistant", "content": "authentication_issue"},
    {"role": "user", "content": "How do I upgrade to the pro plan?"},
    {"role": "assistant", "content": "billing_inquiry"},
    {"role": "user", "content": "Your API is returning 500 errors"},
    {"role": "assistant", "content": "technical_bug"},
    {"role": "user", "content": "Can I get a refund for last month?"},
    {"role": "assistant", "content": "billing_inquiry"},
]


async def classify_intent(message: str) -> str:
    messages = FEW_SHOT_MESSAGES + [{"role": "user", "content": message}]
    async with AsyncOpenAI() as client:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=20,
        )
    return response.choices[0].message.content.strip()


async def self_consistent_answer(question: str, n: int = 5) -> dict:
    """Sample n answers and return the most frequent exact string."""
    async with AsyncOpenAI() as client:
        tasks = []
        for _ in range(n):
            tasks.append(client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer concisely in one sentence."},
                    {"role": "user", "content": question},
                ],
                temperature=0.8,
                max_tokens=100,
            ))

        responses = await asyncio.gather(*tasks)
    answers = [r.choices[0].message.content.strip() for r in responses]

    counter = Counter(answers)
    best_answer, count = counter.most_common(1)[0]
    return {
        "answer": best_answer,
        "agreement": count / n,
        "all_answers": answers,
    }


async def main():
    cot = await chain_of_thought(
        "If a RAG system returns 5 chunks of 512 tokens each, "
        "and the system prompt is 200 tokens, how many tokens are used for context?"
    )
    print(f"CoT: {cot[:200]}...\n")

    intent = await classify_intent("The embeddings endpoint is super slow today")
    print(f"Intent: {intent}\n")

    result = await self_consistent_answer("Is Qdrant or ChromaDB better for production?")
    print(f"Answer ({result['agreement']:.0%} agreement): {result['answer']}")


if __name__ == "__main__":
    asyncio.run(main())
