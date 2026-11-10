"""Pydantic v2 models for an AI service — validation, serialization, and structured output."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: Role
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=128_000)
    stream: bool = False

    @model_validator(mode="after")
    def validate_system_message_first(self):
        if self.messages and self.messages[0].role != Role.SYSTEM:
            pass  # system message is optional
        system_msgs = [m for m in self.messages if m.role == Role.SYSTEM]
        if len(system_msgs) > 1:
            raise ValueError("At most one system message is allowed")
        return self


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float = Field(ge=0.0)


class ChatResponse(BaseModel):
    content: str
    model: str
    usage: TokenUsage
    finish_reason: Literal["stop", "length", "tool_calls"]


class TextAnswer(BaseModel):
    type: Literal["text"] = "text"
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


class CodeAnswer(BaseModel):
    type: Literal["code"] = "code"
    language: str
    code: str
    explanation: str


type LLMAnswer = Annotated[TextAnswer | CodeAnswer, Field(discriminator="type")]


if __name__ == "__main__":
    request = ChatRequest(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Explain async/await in Python."),
        ],
        temperature=0.3,
    )
    print(request.model_dump_json(indent=2))
