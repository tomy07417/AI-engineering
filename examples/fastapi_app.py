"""FastAPI dependency injection, lifespan, and shared AI client example.

Run from the repository root with: uvicorn examples.fastapi_app:app
Requires the project dependencies and OPENAI_API_KEY at application startup.
The /chat endpoint makes real provider requests and may incur costs.
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .settings import Settings, get_settings


class AppState:
    """Shared application state initialized at startup."""

    openai: AsyncOpenAI
    settings: Settings


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up shared resources."""
    settings = get_settings()
    state.settings = settings
    state.openai = AsyncOpenAI(api_key=settings.openai_key_str)
    try:
        yield
    finally:
        await state.openai.close()


app = FastAPI(title="AI Service", lifespan=lifespan)


def get_openai() -> AsyncOpenAI:
    return state.openai


OpenAIDep = Annotated[AsyncOpenAI, Depends(get_openai)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    content: str
    model: str
    tokens_used: int


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, openai: OpenAIDep):
    """Chat endpoint with injected OpenAI client."""
    try:
        response = await openai.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.message}],
            temperature=request.temperature,
        )
    except Exception as exc:
        if "rate_limit" in str(exc).lower():
            raise HTTPException(status_code=429, detail="Rate limit exceeded") from exc
        raise HTTPException(status_code=502, detail="LLM provider error") from exc

    choice = response.choices[0]
    return ChatResponse(
        content=choice.message.content or "",
        model=response.model,
        tokens_used=response.usage.total_tokens if response.usage else 0,
    )


@app.get("/health")
async def health(settings: SettingsDep):
    return {"status": "ok", "environment": settings.environment}
