"""Mocked tests for examples.fastapi_app, without real credentials or API calls.

Run explicitly from the repository root:
    python -m pytest examples/test_fastapi_app.py

Requires project runtime dependencies and the dev extra. ASGITransport does
not start the application lifespan; the OpenAI client and settings are mocked.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from examples.fastapi_app import app, get_settings


@pytest.fixture
def mock_openai_response():
    """Fixture: realistic OpenAI chat completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Mocked AI response", role="assistant"),
                finish_reason="stop",
            )
        ],
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
async def async_client():
    """Provide an isolated async client with settings overridden."""
    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(environment="dev")
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)


class TestChatEndpoint:
    @patch("examples.fastapi_app.state")
    async def test_chat_success(self, mock_state, async_client, mock_openai_response):
        mock_state.openai.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        response = await async_client.post(
            "/chat",
            json={"message": "What is Python?", "model": "gpt-4o-mini"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "content": "Mocked AI response",
            "model": "gpt-4o-mini",
            "tokens_used": 15,
        }
        mock_state.openai.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is Python?"}],
            temperature=0.7,
        )

    @patch("examples.fastapi_app.state")
    async def test_chat_rate_limit(self, mock_state, async_client):
        mock_state.openai.chat.completions.create = AsyncMock(
            side_effect=Exception("rate_limit_exceeded")
        )

        response = await async_client.post("/chat", json={"message": "Hello"})

        assert response.status_code == 429
        assert response.json() == {"detail": "Rate limit exceeded"}
        mock_state.openai.chat.completions.create.assert_awaited_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

    @patch("examples.fastapi_app.state")
    async def test_chat_validation_error(self, mock_state, async_client):
        mock_state.openai.chat.completions.create = AsyncMock()

        response = await async_client.post("/chat", json={"message": ""})

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert len(errors) == 1
        assert errors[0]["loc"] == ["body", "message"]
        assert errors[0]["type"] == "string_too_short"
        mock_state.openai.chat.completions.create.assert_not_awaited()

    async def test_health_endpoint(self, async_client):
        response = await async_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "environment": "dev"}
