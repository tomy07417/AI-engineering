"""SSE streaming endpoint that proxies LLM token generation.

Run from the repository root with: uvicorn examples.sse_streaming:app
Requires project dependencies and OPENAI_API_KEY at startup. Requests to
/chat/stream contact OpenAI and may incur costs. This educational example
returns raw exception messages; sanitize errors before exposing it publicly.
"""

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


client: AsyncOpenAI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the shared client at startup and close it at shutdown."""
    global client
    async with AsyncOpenAI() as client:
        yield


app = FastAPI(lifespan=lifespan)


class StreamRequest(BaseModel):
    message: str = Field(min_length=1)
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."


def sse_event(data: dict) -> str:
    """Format a dict as an SSE event."""
    return f"data: {json.dumps(data)}\n\n"


async def generate_stream(request: StreamRequest) -> AsyncGenerator[str, None]:
    """Yield SSE events and close the provider stream on exit."""
    try:
        stream = await client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.message},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        async with stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield sse_event({
                        "type": "token",
                        "content": chunk.choices[0].delta.content,
                    })

                if chunk.choices and chunk.choices[0].finish_reason:
                    yield sse_event({
                        "type": "finish",
                        "reason": chunk.choices[0].finish_reason,
                    })

                if chunk.usage:
                    yield sse_event({
                        "type": "usage",
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    })

        yield "data: [DONE]\n\n"

    except Exception as exc:
        yield sse_event({"type": "error", "message": str(exc)})
        yield "data: [DONE]\n\n"


@app.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    return StreamingResponse(
        generate_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering.
        },
    )
