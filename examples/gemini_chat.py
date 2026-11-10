"""Google Gemini text generation, image analysis, and embeddings example.

Run: python3 examples/gemini_chat.py
Requires google-genai and GOOGLE_API_KEY in the environment. Running this
example makes real API requests and may incur costs. Model availability and
SDK compatibility must be checked before running. analyze_image expects PNG
bytes and sends the supplied image to Google; main does not call it.
"""

import asyncio
from pathlib import Path

from google import genai


async def chat_gemini(
    prompt: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.7,
) -> dict:
    """Text chat with Gemini."""
    async with genai.Client().aio as client:
        response = await client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=1024,
            ),
        )

    return {
        "content": response.text,
        "model": model,
        "usage": {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        },
    }


async def analyze_image(image_path: str, question: str) -> str:
    """Send a PNG image and a text question to Gemini for analysis."""
    image_bytes = Path(image_path).read_bytes()
    image_part = genai.types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png",
    )

    async with genai.Client().aio as client:
        response = await client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[question, image_part],
        )
    return response.text


async def embed_with_gemini(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using Gemini's embedding model."""
    async with genai.Client().aio as client:
        result = await client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
        )
    return [e.values for e in result.embeddings]


async def main():
    result = await chat_gemini("Explain vector databases in 2 sentences.")
    print(f"Gemini: {result['content']}")
    print(f"Tokens: {result['usage']}\n")

    embeddings = await embed_with_gemini(["Hello world", "AI engineering"])
    print(f"Embedding dim: {len(embeddings[0])}")
    print(f"Vectors: {len(embeddings)}")


if __name__ == "__main__":
    asyncio.run(main())
