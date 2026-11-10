"""Structured outputs with OpenAI and Pydantic validation.

Run: python3 examples/structured_outputs.py
Requires openai, pydantic, and OPENAI_API_KEY. Execution makes real API calls
and may incur costs. Refusals, API errors, and validation failures remain
possible. This example does not implement Anthropic structured outputs.
"""

import asyncio
from enum import StrEnum

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EntityType(StrEnum):
    PERSON = "person"
    COMPANY = "company"
    TECHNOLOGY = "technology"
    LOCATION = "location"


class Entity(BaseModel):
    name: str
    entity_type: EntityType
    relevance: float = Field(ge=0.0, le=1.0)


class TextAnalysis(BaseModel):
    summary: str = Field(description="One-sentence summary")
    sentiment: Sentiment
    entities: list[Entity]
    topics: list[str] = Field(max_length=5)
    language: str
    confidence: float = Field(ge=0.0, le=1.0)


async def analyze_text(text: str) -> TextAnalysis:
    """Parse a structured response, raising on refusal or missing parsed output."""
    async with AsyncOpenAI() as client:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the given text and extract structured information.",
                },
                {"role": "user", "content": text},
            ],
            response_format=TextAnalysis,
            temperature=0,
        )

    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError(f"Refused: {response.choices[0].message.refusal}")
    return result


async def classify_batch(texts: list[str]) -> list[TextAnalysis]:
    """Analyze texts concurrently without a concurrency or rate limit."""
    tasks = [analyze_text(text) for text in texts]
    return await asyncio.gather(*tasks)


async def main():
    text = (
        "OpenAI released GPT-4o in San Francisco, making multimodal AI "
        "accessible to developers worldwide. The model processes text, "
        "images, and audio natively."
    )

    analysis = await analyze_text(text)
    print(f"Summary: {analysis.summary}")
    print(f"Sentiment: {analysis.sentiment}")
    print("Entities:")
    for entity in analysis.entities:
        print(f"  - {entity.name} ({entity.entity_type}): {entity.relevance}")
    print(f"Topics: {analysis.topics}")
    print(f"Confidence: {analysis.confidence}")


if __name__ == "__main__":
    asyncio.run(main())
