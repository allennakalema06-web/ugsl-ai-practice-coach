import pytest
from pydantic import ValidationError

from ugsl_ai_coach.core.config import Settings


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.service_name == "ugsl-ai-practice-coach"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"


def test_environment_overrides_dotenv(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("UGSL_ENVIRONMENT=local\nUGSL_LOG_LEVEL=WARNING\n", encoding="utf-8")
    monkeypatch.setenv("UGSL_SERVICE_NAME", "test-service")
    monkeypatch.setenv("UGSL_ENVIRONMENT", "test")
    settings = Settings(_env_file=env_file)
    assert settings.service_name == "test-service"
    assert settings.environment == "test"
    assert settings.log_level == "WARNING"


def test_invalid_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("UGSL_LOG_LEVEL", "INVALID")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
