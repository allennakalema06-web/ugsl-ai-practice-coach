import pytest

from ugsl_ai_coach.domain.analysis import (
    AnalysisStatus, BodyRegion, Evidence, FindingStatus, SkillCategory,
    StructuredAnalysisResult, StructuredFinding,
)


@pytest.fixture
def make_finding():
    def make(status=FindingStatus.NEEDS_IMPROVEMENT, **overrides):
        values = dict(
            finding_id="F-001", skill=SkillCategory.MOVEMENT, status=status,
            severity=1, confidence=1.0, body_region=BodyRegion.LEFT_HAND,
            start_time_ms=123, end_time_ms=987,
            evidence=None if status == FindingStatus.INSUFFICIENT_EVIDENCE else Evidence(
                expected_value="provisional reference movement", observed_value="measured movement",
                deviation=0.4,
            ),
        )
        values.update(overrides)
        return StructuredFinding(**values)
    return make


@pytest.fixture
def make_analysis(make_finding):
    def make(status=AnalysisStatus.COMPLETED, findings=None, **overrides):
        if findings is None:
            findings = (
                (make_finding(),) if status == AnalysisStatus.COMPLETED else
                (make_finding(FindingStatus.INSUFFICIENT_EVIDENCE),)
                if status == AnalysisStatus.UNANALYZABLE else ()
            )
        values = dict(
            analysis_id="AN-000001", attempt_id="ATT-000001", model_version="fixture-v1",
            status=status, findings=findings,
            overall_score=0.43 if status == AnalysisStatus.COMPLETED else None,
            overall_confidence=1.0 if status != AnalysisStatus.FAILED else None,
        )
        values.update(overrides)
        return StructuredAnalysisResult(**values)
    return make
