"""Typed environment-based service settings."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UGSL_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    service_name: str = Field(default="ugsl-ai-practice-coach", min_length=1)
    environment: str = Field(default="development", min_length=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: SecretStr | None = Field(default=None, repr=False)
    worker_lease_seconds: int = Field(default=300, ge=1, le=86400)
    worker_poll_seconds: float = Field(default=1.0, ge=0.05, le=60, allow_inf_nan=False)
