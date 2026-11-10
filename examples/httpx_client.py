"""Demonstrate httpx connection pooling, retries, and SSE-style streaming.

Requires httpx and tenacity. Running this example contacts OpenAI using the
placeholder key below. POST retries can duplicate processing or costs if the
provider already processed a request; review provider idempotency guarantees
before using this pattern in production.
"""

import asyncio

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout))


class AIClient:
    """Async HTTP client demonstrating pooled AI API calls."""

    def __init__(self, base_url: str, api_key: str):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,  # LLM generation can be slow.
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception(is_retryable),
    )
    async def post_json(self, path: str, payload: dict) -> dict:
        """POST JSON with automatic retries on selected transient failures."""
        response = await self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    async def post_stream(self, path: str, payload: dict):
        """POST with SSE-style streaming, yielding single-line data payloads."""
        async with self.client.stream("POST", path, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield data

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


async def main():
    async with AIClient("https://api.openai.com/v1", "sk-...") as client:
        result = await client.post_json(
            "/chat/completions",
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello!"}],
            },
        )
        print(result["choices"][0]["message"]["content"])

        async for chunk in client.post_stream(
            "/chat/completions",
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        ):
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
