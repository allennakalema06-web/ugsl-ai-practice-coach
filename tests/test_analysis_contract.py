import json

import pytest
from pydantic import ValidationError

from ugsl_ai_coach.domain.analysis import (
    AnalysisStatus, BodyRegion, Evidence, FindingStatus, SkillCategory,
    StructuredAnalysisResult, StructuredFinding,
)


@pytest.fixture
def finding_payload():
    return {
        "finding_id": "F-001", "skill": "MOVEMENT", "status": "NEEDS_IMPROVEMENT",
        "severity": 2, "confidence": 0.91, "body_region": "RIGHT_HAND",
        "start_time_ms": 3420, "end_time_ms": 4180,
        "evidence": {"expected_value": "UPWARD", "observed_value": "OUTWARD", "deviation": 0.21},
    }


@pytest.fixture
def analysis_payload(finding_payload):
    return {
        "analysis_id": "AN-014902", "attempt_id": "ATT-009418",
        "model_version": "cv-engine-v1.3.0", "status": "COMPLETED",
        "overall_score": 0.72, "overall_confidence": 0.94,
        "findings": [finding_payload, {
            "finding_id": "F-002", "skill": "HANDSHAPE", "status": "STRONG",
            "severity": 0, "confidence": 0.94, "body_region": "RIGHT_HAND",
            "start_time_ms": 1200, "end_time_ms": 4180,
            "evidence": {"expected_value": "CLOSED_FIST_THUMB_UP", "observed_value": "CLOSED_FIST_THUMB_UP", "deviation": 0.04},
        }],
    }


def test_completed_example_round_trips(analysis_payload):
    result = StructuredAnalysisResult.model_validate(analysis_payload)
    assert result.findings[0].skill == SkillCategory.MOVEMENT
    assert result.findings[0].evidence.expected_value == "UPWARD"
    assert result.findings[0].evidence.observed_value == "OUTWARD"
    assert json.loads(result.model_dump_json()) == analysis_payload
    assert StructuredAnalysisResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("field,value", [
    ("analysis_id", "AN-12345"), ("analysis_id", "AN-abcdef"), ("analysis_id", "XAN-014902"),
    ("attempt_id", "ATT-12345"), ("attempt_id", "ATT-abcdef"), ("attempt_id", "ATT-009418x"),
    ("status", "PENDING"), ("model_version", " \t\n"),
    ("overall_score", -0.01), ("overall_score", 1.01),
    ("overall_confidence", -0.01), ("overall_confidence", 1.01),
])
def test_invalid_analysis_fields(analysis_payload, field, value):
    analysis_payload[field] = value
    with pytest.raises(ValidationError):
        StructuredAnalysisResult.model_validate(analysis_payload)


@pytest.mark.parametrize("field,value", [
    ("finding_id", "F-01"), ("finding_id", "F-abc"), ("finding_id", "F-001x"),
    ("skill", "DIRECTION"), ("skill", "UNKNOWN"), ("body_region", "FOOT"),
    ("status", "GOOD"), ("confidence", -0.01), ("confidence", 1.01), ("confidence", 91),
    ("severity", -1), ("severity", 4), ("severity", 1.5), ("severity", True), ("severity", "2"),
    ("start_time_ms", -1), ("end_time_ms", -1), ("end_time_ms", 3419),
    ("start_time_ms", 1.5), ("start_time_ms", True), ("end_time_ms", "4180"),
])
def test_invalid_finding_fields(finding_payload, field, value):
    finding_payload[field] = value
    with pytest.raises(ValidationError):
        StructuredFinding.model_validate(finding_payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "0.5", True])
@pytest.mark.parametrize("field", ["overall_score", "overall_confidence", "confidence", "deviation"])
def test_invalid_numeric_values(analysis_payload, field, value):
    if field in ("overall_score", "overall_confidence"):
        analysis_payload[field] = value
    elif field == "confidence":
        analysis_payload["findings"][0][field] = value
    else:
        analysis_payload["findings"][0]["evidence"][field] = value
    with pytest.raises(ValidationError):
        StructuredAnalysisResult.model_validate(analysis_payload)


@pytest.mark.parametrize("status", ["STRONG", "ACCEPTABLE", "NEEDS_IMPROVEMENT", "WARNING"])
@pytest.mark.parametrize("omit", [True, False])
def test_judgment_requires_evidence(finding_payload, status, omit):
    finding_payload["status"] = status
    if omit:
        del finding_payload["evidence"]
    else:
        finding_payload["evidence"] = None
    with pytest.raises(ValidationError):
        StructuredFinding.model_validate(finding_payload)


@pytest.mark.parametrize("omit", [True, False])
def test_insufficient_evidence_can_omit_evidence(finding_payload, omit):
    finding_payload["status"] = "INSUFFICIENT_EVIDENCE"
    if omit:
        del finding_payload["evidence"]
    else:
        finding_payload["evidence"] = None
    assert StructuredFinding.model_validate(finding_payload).evidence is None


@pytest.mark.parametrize("status", ["COMPLETED", "UNANALYZABLE"])
@pytest.mark.parametrize("omit", [True, False])
def test_required_terminal_values(analysis_payload, status, omit):
    analysis_payload.update(status=status, findings=[])
    if status == "UNANALYZABLE":
        analysis_payload.pop("overall_score")
    if omit:
        analysis_payload.pop("overall_confidence")
    else:
        analysis_payload["overall_confidence"] = None
    with pytest.raises(ValidationError):
        StructuredAnalysisResult.model_validate(analysis_payload)


@pytest.mark.parametrize("omit", [True, False])
def test_completed_requires_score(analysis_payload, omit):
    if omit:
        del analysis_payload["overall_score"]
    else:
        analysis_payload["overall_score"] = None
    with pytest.raises(ValidationError):
        StructuredAnalysisResult.model_validate(analysis_payload)


@pytest.mark.parametrize("status", ["UNANALYZABLE", "FAILED"])
def test_noncompleted_rejects_score(analysis_payload, status):
    analysis_payload.update(status=status, findings=[])
    with pytest.raises(ValidationError):
        StructuredAnalysisResult.model_validate(analysis_payload)


@pytest.mark.parametrize("status", ["STRONG", "ACCEPTABLE", "NEEDS_IMPROVEMENT", "WARNING"])
def test_abstention_rejects_judgments(analysis_payload, status):
    analysis_payload.update(status="UNANALYZABLE", overall_score=None)
    analysis_payload["findings"] = [dict(analysis_payload["findings"][0], status=status)]
    with pytest.raises(ValidationError):
        StructuredAnalysisResult.model_validate(analysis_payload)


@pytest.mark.parametrize("with_findings", [True, False])
def test_valid_abstention(analysis_payload, with_findings):
    analysis_payload.update(status="UNANALYZABLE", overall_score=None)
    finding = dict(analysis_payload["findings"][0], status="INSUFFICIENT_EVIDENCE", evidence=None)
    analysis_payload["findings"] = [finding] if with_findings else []
    result = StructuredAnalysisResult.model_validate(analysis_payload)
    assert result.overall_score is None
    assert result.overall_confidence == 0.94


@pytest.mark.parametrize("confidence", [None, 0.0, 1.0])
def test_valid_failure(analysis_payload, confidence):
    analysis_payload.update(status="FAILED", overall_score=None, overall_confidence=confidence, findings=[])
    assert StructuredAnalysisResult.model_validate(analysis_payload).findings == ()


def test_failure_may_omit_score_and_confidence(analysis_payload):
    analysis_payload.update(status="FAILED", findings=[])
    del analysis_payload["overall_score"]
    del analysis_payload["overall_confidence"]
    result = StructuredAnalysisResult.model_validate(analysis_payload)
    assert result.overall_score is None and result.overall_confidence is None


@pytest.mark.parametrize("status", ["NEEDS_IMPROVEMENT", "INSUFFICIENT_EVIDENCE"])
def test_failure_rejects_findings(analysis_payload, status):
    analysis_payload.update(status="FAILED", overall_score=None)
    analysis_payload["findings"] = [dict(analysis_payload["findings"][0], status=status)]
    with pytest.raises(ValidationError):
        StructuredAnalysisResult.model_validate(analysis_payload)


@pytest.mark.parametrize("location", ["analysis", "finding", "evidence"])
def test_extra_fields_rejected(analysis_payload, location):
    target = analysis_payload
    if location != "analysis":
        target = target["findings"][0]
    if location == "evidence":
        target = target["evidence"]
    target["typo"] = "unexpected"
    with pytest.raises(ValidationError):
        StructuredAnalysisResult.model_validate(analysis_payload)


@pytest.mark.parametrize("value", [0, 1])
def test_normalized_boundaries_and_empty_completed(analysis_payload, value):
    analysis_payload.update(overall_score=value, overall_confidence=value, findings=[])
    assert StructuredAnalysisResult.model_validate(analysis_payload).overall_score == value


@pytest.mark.parametrize("deviation", [-27.5, 0, 180, 12345.6])
def test_deviation_has_no_arbitrary_range(finding_payload, deviation):
    finding_payload["evidence"]["deviation"] = deviation
    assert StructuredFinding.model_validate(finding_payload).evidence.deviation == deviation


def test_trimmed_version_and_long_identifiers(analysis_payload):
    analysis_payload.update(model_version="  custom engine  ", analysis_id="AN-1234567", attempt_id="ATT-1234567")
    analysis_payload["findings"][0]["finding_id"] = "F-1234"
    assert StructuredAnalysisResult.model_validate(analysis_payload).model_version == "custom engine"


def test_equal_timestamps_and_explicit_severity(finding_payload):
    finding_payload.update(status="STRONG", severity=3, start_time_ms=0, end_time_ms=0)
    assert StructuredFinding.model_validate(finding_payload).severity == 3


def test_nested_contract_immutability(analysis_payload):
    result = StructuredAnalysisResult.model_validate(analysis_payload)
    for obj, field, value in [
        (result, "overall_score", 0.1),
        (result.findings[0], "severity", 3),
        (result.findings[0].evidence, "deviation", 100),
    ]:
        with pytest.raises(ValidationError):
            setattr(obj, field, value)
    with pytest.raises(TypeError):
        result.findings[0] = result.findings[1]
    assert not hasattr(result.findings, "append")
    assert json.loads(result.model_dump_json()) == analysis_payload


def test_exact_taxonomies():
    assert {v.value for v in AnalysisStatus} == {"COMPLETED", "FAILED", "UNANALYZABLE"}
    assert {v.value for v in SkillCategory} == {"HANDSHAPE", "ORIENTATION", "LOCATION", "MOVEMENT", "TIMING", "SEQUENCE", "MOVEMENT_RANGE", "BODY_POSITION"}
    assert {v.value for v in FindingStatus} == {"STRONG", "ACCEPTABLE", "NEEDS_IMPROVEMENT", "WARNING", "INSUFFICIENT_EVIDENCE"}
    assert {v.value for v in BodyRegion} == {"LEFT_HAND", "RIGHT_HAND", "BOTH_HANDS", "LEFT_ARM", "RIGHT_ARM", "HEAD", "UPPER_BODY"}
