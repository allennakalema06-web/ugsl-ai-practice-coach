import pytest


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch):
    for name in ("UGSL_SERVICE_NAME", "UGSL_ENVIRONMENT", "UGSL_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
