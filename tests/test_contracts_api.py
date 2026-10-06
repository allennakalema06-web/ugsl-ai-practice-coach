from fastapi.testclient import TestClient

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.main import create_app


def test_schema_endpoint_exposes_authoritative_model():
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get("/api/v1/contracts/analysis")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.json() == StructuredAnalysisResult.model_json_schema()
        assert response.json()["title"] == "StructuredAnalysisResult"
        assert response.json()["properties"]["findings"]["type"] == "array"
        assert response.json()["additionalProperties"] is False
        assert client.get("/api/v1/health").json() == {"status": "ok", "service": "ugsl-ai-practice-coach"}
        assert "/api/v1/contracts/analysis" in client.get("/openapi.json").json()["paths"]
        assert client.post("/api/v1/analyze").status_code == 404
