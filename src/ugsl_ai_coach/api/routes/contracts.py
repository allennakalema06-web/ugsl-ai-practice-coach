"""Inspect the authoritative contract without performing analysis."""

from typing import Any

from fastapi import APIRouter

from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult

router = APIRouter(prefix="/contracts", tags=["contracts"])


@router.get("/analysis", summary="Structured Findings Contract v1 JSON Schema")
def analysis_schema() -> dict[str, Any]:
    return StructuredAnalysisResult.model_json_schema()
