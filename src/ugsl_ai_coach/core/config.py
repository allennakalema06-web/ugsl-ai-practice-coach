"""Typed environment-based service settings."""

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UGSL_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    service_name: str = Field(default="ugsl-ai-practice-coach", min_length=1)
    environment: str = Field(default="development", min_length=1)
    api_docs_enabled: bool | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    database_url: SecretStr | None = Field(default=None, repr=False)
    worker_lease_seconds: int = Field(default=300, ge=1, le=86400)
    worker_poll_seconds: float = Field(default=1.0, ge=0.05, le=60, allow_inf_nan=False)
    service_token: SecretStr | None = Field(default=None, repr=False)
    object_store_bucket: str | None = None
    object_store_region: str | None = None
    object_store_endpoint_url: str | None = None
    learner_video_prefix: str = "learner-videos/"
    reference_profile_prefix: str = "reference-profiles/"
    max_video_bytes: int = Field(default=50 * 1024 * 1024, ge=1, le=1024 * 1024 * 1024)
    max_reference_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    hand_model_path: str | None = None
    pose_model_path: str | None = None
    coaching_lease_seconds: int = Field(default=300, ge=1, le=86400)
    coaching_poll_seconds: float = Field(default=1.0, ge=0.05, le=60, allow_inf_nan=False)
    coaching_retry_base_seconds: int = Field(default=5, ge=1, le=86400)
    coaching_retry_max_seconds: int = Field(default=300, ge=1, le=86400)
    submit_rate_limit_per_minute: int = Field(default=60, ge=1, le=1000000)
    read_rate_limit_per_minute: int = Field(default=600, ge=1, le=1000000)
    temp_media_max_age_seconds: int = Field(default=86400, ge=3600, le=604800)

    @model_validator(mode="after")
    def retry_bounds(self):
        if self.coaching_retry_max_seconds < self.coaching_retry_base_seconds:
            raise ValueError("Coaching retry maximum must be at least the base")
        return self
