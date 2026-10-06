import json

import pytest
from pydantic import ValidationError

from ugsl_ai_coach.comparison.engine import compare_movement
from ugsl_ai_coach.comparison.models import ComparisonRequest, HandCorrespondence, InvalidComparisonInput
from ugsl_ai_coach.comparison.policy import ComparisonPolicy
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult


def test_completed_traceability_and_determinism(make_extraction, comparison_request):
    reference, learner = make_extraction(), make_extraction()
    before = (reference.model_dump_json(), learner.model_dump_json())
    outcome = compare_movement(reference, learner, comparison_request)
    assert outcome == compare_movement(reference, learner, comparison_request)
    assert outcome.analysis.status == "COMPLETED"
    assert outcome.analysis.overall_score == outcome.analysis.overall_confidence == 1
    finding = outcome.analysis.findings[0]
    assert finding.skill == "MOVEMENT" and finding.status == "STRONG"
    assert finding.finding_id == outcome.metrics.finding_id
    assert finding.evidence.deviation == outcome.metrics.dtw_mean_path_distance == 0
    assert finding.start_time_ms == 0 and finding.end_time_ms == 363
    assert outcome.alignment.path[0].learner_frame_index == outcome.learner_trajectory.usable_points[0].frame_index
    assert outcome.request.reference_id == "synthetic-reference"
    assert outcome.policy.policy_version in outcome.analysis.model_version
    assert StructuredAnalysisResult.model_validate_json(outcome.analysis.model_dump_json()) == outcome.analysis
    json.dumps(outcome.model_dump(mode="json"), allow_nan=False)
    assert (reference.model_dump_json(), learner.model_dump_json()) == before


def test_low_similarity_can_have_high_independent_confidence(make_extraction, comparison_request):
    reference = make_extraction()
    identical = compare_movement(reference, reference, comparison_request)
    different = compare_movement(reference, make_extraction(offset=10), comparison_request)
    assert different.analysis.overall_score < 0.1
    assert identical.analysis.overall_confidence == different.analysis.overall_confidence == 1
    assert different.analysis.findings[0].status == "NEEDS_IMPROVEMENT"
    assert {f.skill for f in different.analysis.findings} == {"MOVEMENT"}
    assert "UPWARD" not in different.analysis.findings[0].evidence.model_dump_json()


@pytest.mark.parametrize("side", ["learner", "reference", "both"])
def test_low_coverage_abstains_without_performance_score(make_extraction, comparison_request, side):
    full, sparse = make_extraction(), make_extraction(missing=(0, 1, 2))
    reference = sparse if side in ("reference", "both") else full
    learner = sparse if side in ("learner", "both") else full
    outcome = compare_movement(reference, learner, comparison_request)
    assert outcome.analysis.status == "UNANALYZABLE"
    assert outcome.analysis.overall_score is None
    assert outcome.analysis.overall_confidence == 0.7
    assert outcome.alignment is None and outcome.metrics is None
    assert all(f.status == "INSUFFICIENT_EVIDENCE" and f.severity == 0 and f.evidence is None for f in outcome.analysis.findings)
    assert outcome.reason == f"{side.upper()}_INSUFFICIENT"
    StructuredAnalysisResult.model_validate_json(outcome.analysis.model_dump_json())


def test_no_learner_observations_means_no_fabricated_interval(make_extraction, comparison_request):
    outcome = compare_movement(make_extraction(), make_extraction(missing=range(10)), comparison_request)
    assert outcome.analysis.status == "UNANALYZABLE"
    assert outcome.analysis.overall_confidence == 0
    assert not outcome.analysis.findings


def test_correspondence_required(make_extraction):
    comparison_request = ComparisonRequest(analysis_id="AN-000001", attempt_id="ATT-000001", reference_id="reference")
    outcome = compare_movement(make_extraction(), make_extraction(), comparison_request)
    assert outcome.reason == "CORRESPONDENCE_UNESTABLISHED"
    assert outcome.analysis.status == "UNANALYZABLE" and outcome.analysis.overall_score is None
    assert not outcome.analysis.findings and outcome.alignment is None


def test_no_automatic_hand_swap_but_explicit_pairing_allowed(make_extraction, comparison_request):
    reference, learner = make_extraction(label="Left"), make_extraction(label="Right")
    assert compare_movement(reference, learner, comparison_request).analysis.status == "UNANALYZABLE"
    explicit = ComparisonRequest(analysis_id=comparison_request.analysis_id, attempt_id=comparison_request.attempt_id, reference_id=comparison_request.reference_id,
        correspondence=HandCorrespondence(reference_label="Left", learner_label="Right", body_region="LEFT_HAND",
                                          establishment_basis="Caller provided synthetic anatomical calibration"))
    assert compare_movement(reference, learner, explicit).analysis.overall_score == 1


def test_no_mirroring_or_hand_start_recentering(make_extraction, comparison_request):
    path = [(i / 10, 0) for i in range(10)]
    reference = make_extraction(path)
    mirrored = compare_movement(reference, make_extraction([(-x, y) for x, y in path]), comparison_request)
    shifted = compare_movement(reference, make_extraction(path, offset=1), comparison_request)
    assert mirrored.metrics.dtw_mean_path_distance > 0
    assert shifted.metrics.dtw_mean_path_distance > 0


def test_resource_failure_is_failed_not_learner_judgment(make_extraction, comparison_request):
    outcome = compare_movement(make_extraction(), make_extraction(), comparison_request, ComparisonPolicy(maximum_alignment_cells=99))
    assert outcome.analysis.status == "FAILED"
    assert outcome.reason == "ALIGNMENT_RESOURCE_LIMIT"
    assert outcome.analysis.overall_score is None and outcome.analysis.overall_confidence is None
    assert not outcome.analysis.findings
    StructuredAnalysisResult.model_validate_json(outcome.analysis.model_dump_json())


def test_numerical_failure_is_failed_without_nonfinite_values(make_extraction, comparison_request):
    outcome = compare_movement(make_extraction([(1e308, 0)] * 10), make_extraction([(-1e308, 0)] * 10), comparison_request)
    assert outcome.analysis.status == "FAILED" and outcome.reason == "NUMERICAL_FAILURE"
    json.dumps(outcome.model_dump(mode="json"), allow_nan=False)


def test_programmer_errors_are_not_masked(make_extraction, comparison_request, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("programming fault")
    monkeypatch.setattr("ugsl_ai_coach.comparison.engine.align_dtw", broken)
    with pytest.raises(RuntimeError, match="programming fault"):
        compare_movement(make_extraction(), make_extraction(), comparison_request)


def test_config_changes_classification_without_algorithm_changes(make_extraction, comparison_request):
    reference, learner = make_extraction([(0, 0)] * 10), make_extraction([(0.2, 0)] * 10)
    default = compare_movement(reference, learner, comparison_request)
    custom = compare_movement(reference, learner, comparison_request, ComparisonPolicy(
        policy_version="test-engineering-v2", strong_maximum_distance=0.25, acceptable_maximum_distance=0.5))
    assert default.analysis.findings[0].status == "ACCEPTABLE"
    assert custom.analysis.findings[0].status == "STRONG"
    assert default.alignment == custom.alignment
    assert default.analysis.overall_score == custom.analysis.overall_score


def test_invalid_caller_input_is_not_a_fake_analysis(make_extraction, comparison_request):
    with pytest.raises(InvalidComparisonInput):
        compare_movement(None, make_extraction(), comparison_request)
    with pytest.raises(ValidationError):
        HandCorrespondence(reference_label="Left", learner_label="Left", body_region="LEFT_HAND", establishment_basis=" ")


def test_outputs_are_frozen(make_extraction, comparison_request):
    outcome = compare_movement(make_extraction(), make_extraction(), comparison_request)
    with pytest.raises(ValidationError):
        outcome.reason = "changed"
    with pytest.raises(ValidationError):
        outcome.policy.policy_version = "changed"


def test_actual_m3_normalization_removes_camera_translation_and_scale(make_extraction, comparison_request):
    from ugsl_ai_coach.cv.models import Coordinates, CoverageSummary, ExtractionResult, FrameObservation, HandObservation, Landmark, RawObservation
    from ugsl_ai_coach.cv.normalization import normalize
    base = make_extraction()
    def normalized_extraction(dx, dy, scale):
        frames = []
        for frame in base.frames:
            def transform(p):
                return Landmark(index=p.index, coordinates=Coordinates(
                    x=p.coordinates.x * scale + dx, y=p.coordinates.y * scale + dy, z=p.coordinates.z))
            shoulders = tuple(transform(Landmark(index=i, coordinates=Coordinates(x=x, y=0.5)))
                              for i, x in [(11, 0.25), (12, 0.75)])
            hand = frame.raw.hands[0]
            raw = RawObservation(pose=shoulders, hands=(HandObservation(
                reported_handedness=hand.reported_handedness, handedness_confidence=hand.handedness_confidence,
                landmarks=tuple(transform(p) for p in hand.landmarks)),))
            frames.append(FrameObservation(frame_index=frame.frame_index, timestamp_ms=frame.timestamp_ms,
                raw=raw, normalization=normalize(raw, 100, 100, base.normalization_config)))
        return ExtractionResult(video=base.video, sampling=base.sampling, normalization_config=base.normalization_config,
            frames=tuple(frames), coverage=CoverageSummary.from_frames(frames))
    reference, moved = normalized_extraction(0, 0, 1), normalized_extraction(0.3, -0.1, 0.7)
    assert reference.frames[0].raw != moved.frames[0].raw
    outcome = compare_movement(reference, moved, comparison_request)
    assert outcome.metrics.dtw_mean_path_distance == pytest.approx(0, abs=1e-12)
    assert outcome.analysis.overall_score == pytest.approx(1)


def test_only_normalized_wrist_xy_affects_movement(make_extraction, comparison_request):
    from ugsl_ai_coach.cv.models import ExtractionResult
    reference = make_extraction()
    altered = reference.model_dump()
    for frame in altered["frames"]:
        for point in frame["normalization"]["hands"][0]:
            if point["index"] != 0:
                point["coordinates"]["x"] += 1000
        for point in frame["raw"]["hands"][0]["landmarks"]:
            point["coordinates"]["z"] = -1234.0
    outcome = compare_movement(reference, ExtractionResult.model_validate(altered), comparison_request)
    assert outcome.analysis.overall_score == 1


def test_engine_aligns_different_durations_without_timing_findings(make_extraction, comparison_request):
    reference = make_extraction([(0, 0), (1, 0), (2, 0)], times=[0, 50, 100])
    learner = make_extraction([(0, 0), (0, 0), (1, 0), (1, 0), (2, 0), (2, 0)], times=[0, 50, 100, 150, 200, 250])
    outcome = compare_movement(reference, learner, comparison_request, ComparisonPolicy(minimum_usable_observations=2))
    assert outcome.metrics.dtw_mean_path_distance == 0
    assert outcome.metrics.reference.observed_duration_ms == 100
    assert outcome.metrics.learner.observed_duration_ms == 250
    assert {f.skill for f in outcome.analysis.findings} == {"MOVEMENT"}


def test_eligible_sparse_paths_retain_alignment_to_actual_frames(make_extraction, comparison_request):
    source = make_extraction(missing=(1, 2))
    outcome = compare_movement(source, source, comparison_request)
    assert outcome.analysis.status == "COMPLETED"
    assert outcome.analysis.overall_score == 1
    assert outcome.analysis.overall_confidence == 0.8
    assert {p.learner_frame_index for p in outcome.alignment.path} == {0, 3, 4, 5, 6, 7, 8, 9}


def test_reference_eligibility_threshold_can_be_replaced(make_extraction, comparison_request):
    reference, learner = make_extraction(missing=(0, 1, 2)), make_extraction()
    assert compare_movement(reference, learner, comparison_request).analysis.status == "UNANALYZABLE"
    outcome = compare_movement(reference, learner, comparison_request, ComparisonPolicy(
        policy_version="lower-coverage-engineering-test", minimum_usable_observations=7, minimum_reference_coverage=0.7))
    assert outcome.analysis.status == "COMPLETED"
    assert outcome.analysis.overall_confidence == 0.7
