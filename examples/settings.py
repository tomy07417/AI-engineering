"""Type-safe configuration with pydantic-settings for AI services."""

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    environment: Environment = Environment.DEV

    # SecretStr masks normal representations; it does not encrypt credentials.
    openai_api_key: SecretStr
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # Model defaults remain the same across environments unless overridden.
    default_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Vector DB.
    chroma_persist_dir: str = "./chroma_db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    # Service config.
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PROD

    @property
    def openai_key_str(self) -> str:
        """Expose the raw key for API calls; never log this value."""
        return self.openai_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Load settings on first call and cache them for subsequent calls."""
    return Settings()


if __name__ == "__main__":
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"Model: {settings.default_model}")
    print(f"OpenAI key: {settings.openai_api_key}")  # Prints a masked value.
    print(f"Production: {settings.is_production}")
