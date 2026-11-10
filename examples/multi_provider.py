"""Unified chat interface for OpenAI and Anthropic, with model routing.

Run: python3 examples/multi_provider.py
Requires openai, anthropic, OPENAI_API_KEY, and ANTHROPIC_API_KEY. Execution
makes real API requests and may incur costs. This example does not implement
Gemini. The router initializes both providers, so both credentials are needed.
Anthropic embeddings are unsupported. Provider message formats and model
capabilities are not interchangeable; this example handles basic text only.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

import anthropic
from openai import AsyncOpenAI


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResult:
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self, messages: list[ChatMessage], model: str, temperature: float = 0.7
    ) -> ChatResult:
        ...

    @abstractmethod
    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI()

    async def close(self):
        await self.client.close()

    async def chat(self, messages, model="gpt-4o-mini", temperature=0.7):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        resp = await self.client.chat.completions.create(
            model=model, messages=msgs, temperature=temperature,
        )
        return ChatResult(
            content=resp.choices[0].message.content or "",
            model=resp.model,
            provider="openai",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )

    async def embed(self, texts, model="text-embedding-3-small"):
        resp = await self.client.embeddings.create(input=texts, model=model)
        return [e.embedding for e in resp.data]


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.AsyncAnthropic()

    async def close(self):
        await self.client.close()

    async def chat(self, messages, model="claude-sonnet-4-20250514", temperature=0.7):
        system = ""
        api_msgs = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                api_msgs.append({"role": m.role, "content": m.content})

        resp = await self.client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=api_msgs,
            temperature=temperature,
        )
        return ChatResult(
            content=resp.content[0].text,
            model=resp.model,
            provider="anthropic",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    async def embed(self, texts, model=""):
        raise NotImplementedError("Anthropic does not offer embeddings")


class LLMRouter:
    """Route requests using an explicit model-to-provider map."""

    def __init__(self):
        self.providers: dict[str, LLMProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
        }
        self.model_map: dict[str, str] = {
            "gpt-4o": "openai",
            "gpt-4o-mini": "openai",
            "claude-sonnet-4-20250514": "anthropic",
        }

    async def close(self):
        """Attempt to close both concrete provider clients."""
        try:
            await self.providers["openai"].close()
        finally:
            await self.providers["anthropic"].close()

    async def chat(self, messages: list[ChatMessage], model: str, **kwargs) -> ChatResult:
        provider_name = self.model_map.get(model)
        if not provider_name:
            raise ValueError(f"Unknown model: {model}")
        provider = self.providers[provider_name]
        return await provider.chat(messages, model, **kwargs)


async def main():
    router = LLMRouter()
    try:
        messages = [
            ChatMessage("system", "Be concise."),
            ChatMessage("user", "What is AI Engineering?"),
        ]

        for model in ["gpt-4o-mini", "claude-sonnet-4-20250514"]:
            result = await router.chat(messages, model)
            print(f"[{result.provider}] {result.content[:100]}...")
            print(f"  Tokens: {result.input_tokens} + {result.output_tokens}\n")
    finally:
        await router.close()


if __name__ == "__main__":
    asyncio.run(main())
