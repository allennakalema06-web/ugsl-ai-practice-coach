from fastapi import FastAPI
from fastapi.testclient import TestClient

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.main import create_app


def test_application_can_be_created():
    app = create_app(Settings(_env_file=None))
    assert isinstance(app, FastAPI)
    assert app.title == "UgSL AI Practice Coach"
    assert app.version == "0.1.0"


def test_health_response():
    settings = Settings(_env_file=None, service_name="ugsl-ai-practice-coach")
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {"status": "ok", "service": "ugsl-ai-practice-coach"}
        assert client.get("/api/v1/health").json() == response.json()


def test_health_uses_application_settings():
    with TestClient(create_app(Settings(_env_file=None, service_name="test-coach"))) as client:
        assert client.get("/api/v1/health").json()["service"] == "test-coach"
