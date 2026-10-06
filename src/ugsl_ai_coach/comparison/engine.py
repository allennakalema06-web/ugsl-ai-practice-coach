"""Reference comparison produces provisional MOVEMENT evidence, never coaching."""

import math

from ugsl_ai_coach.cv.models import ExtractionResult
from ugsl_ai_coach.domain.analysis import (
    AnalysisStatus, Evidence, FindingStatus, SkillCategory, StructuredAnalysisResult, StructuredFinding,
)
from ugsl_ai_coach.comparison.dtw import align_dtw
from ugsl_ai_coach.comparison.metrics import check_eligibility, movement_similarity
from ugsl_ai_coach.comparison.models import (
    AlignmentResourceLimit, AlignmentResult, ComparisonMetrics, ComparisonModel, ComparisonRequest,
    EligibilityReport, ExtractionContext, InvalidComparisonInput, MovementTrajectory, NumericalComparisonFailure, OutcomeReason,
)
from ugsl_ai_coach.comparison.policy import ComparisonPolicy
from ugsl_ai_coach.comparison.trajectory import select_trajectory


class ComparisonOutcome(ComparisonModel):
    analysis: StructuredAnalysisResult
    request: ComparisonRequest
    policy: ComparisonPolicy
    reference_context: ExtractionContext
    learner_context: ExtractionContext
    reason: OutcomeReason | None = None
    reference_trajectory: MovementTrajectory | None = None
    learner_trajectory: MovementTrajectory | None = None
    eligibility: EligibilityReport | None = None
    alignment: AlignmentResult | None = None
    metrics: ComparisonMetrics | None = None


def _interval(trajectory: MovementTrajectory) -> tuple[int, int] | None:
    points = trajectory.usable_points
    if not points:
        return None
    # Enclose actual evidence timestamps; M2 requires integer milliseconds.
    return math.floor(points[0].timestamp_ms), math.ceil(points[-1].timestamp_ms)


def compare_movement(reference: ExtractionResult, learner: ExtractionResult,
                     request: ComparisonRequest, policy: ComparisonPolicy | None = None) -> ComparisonOutcome:
    if not isinstance(reference, ExtractionResult) or not isinstance(learner, ExtractionResult):
        raise InvalidComparisonInput("Reference and learner must be M3 ExtractionResults")
    if not isinstance(request, ComparisonRequest) or (policy is not None and not isinstance(policy, ComparisonPolicy)):
        raise InvalidComparisonInput("Use validated ComparisonRequest and ComparisonPolicy inputs")
    policy = policy if policy is not None else ComparisonPolicy()
    reference_context = ExtractionContext.from_extraction(reference)
    ref = None if request.correspondence is None else select_trajectory(reference, request.correspondence.reference_label, policy)
    return _compare_selected(ref, reference_context, learner, request, policy)


def compare_precomputed_movement(reference: MovementTrajectory, reference_context: ExtractionContext,
                                learner: ExtractionResult, request: ComparisonRequest,
                                policy: ComparisonPolicy | None = None) -> ComparisonOutcome:
    """Same M4 algorithm with an already-selected, validated expert trajectory."""
    if not isinstance(reference, MovementTrajectory) or not isinstance(reference_context, ExtractionContext):
        raise InvalidComparisonInput("Use typed precomputed reference trajectory and context")
    if not isinstance(learner, ExtractionResult) or not isinstance(request, ComparisonRequest):
        raise InvalidComparisonInput("Use typed learner extraction and comparison request")
    if policy is not None and not isinstance(policy, ComparisonPolicy):
        raise InvalidComparisonInput("Use a typed comparison policy")
    reference = MovementTrajectory.model_validate(reference.model_dump(mode="python"))
    reference_context = ExtractionContext.model_validate(reference_context.model_dump(mode="python"))
    if request.correspondence is not None and reference.reported_label != request.correspondence.reference_label:
        raise InvalidComparisonInput("Precomputed reference must match established correspondence")
    return _compare_selected(reference, reference_context, learner, request, policy or ComparisonPolicy())


def _compare_selected(ref, reference_context, learner, request, policy):
    version = f"{policy.algorithm_version}/{policy.policy_version}"
    context = dict(reference_context=reference_context, learner_context=ExtractionContext.from_extraction(learner))

    def analysis(status, confidence=None, score=None, findings=()):
        return StructuredAnalysisResult(analysis_id=request.analysis_id, attempt_id=request.attempt_id,
                                        model_version=version, status=status, overall_score=score,
                                        overall_confidence=confidence, findings=findings)

    if request.correspondence is None:
        return ComparisonOutcome(analysis=analysis(AnalysisStatus.UNANALYZABLE, 0.0), request=request,
                                 policy=policy, reason=OutcomeReason.CORRESPONDENCE_UNESTABLISHED, **context)
    learn = select_trajectory(learner, request.correspondence.learner_label, policy)
    eligibility = check_eligibility(ref, learn, policy)
    artifacts = dict(request=request, policy=policy, reference_trajectory=ref,
                     learner_trajectory=learn, eligibility=eligibility, **context)
    interval = _interval(learn)
    if not eligibility.reference_eligible or not eligibility.learner_eligible:
        reason = (OutcomeReason.BOTH_INSUFFICIENT if not eligibility.reference_eligible and not eligibility.learner_eligible
                  else OutcomeReason.REFERENCE_INSUFFICIENT if not eligibility.reference_eligible
                  else OutcomeReason.LEARNER_INSUFFICIENT)
        findings = ()
        if interval is not None:
            findings = (StructuredFinding(
                finding_id=request.finding_id, skill=SkillCategory.MOVEMENT,
                status=FindingStatus.INSUFFICIENT_EVIDENCE, severity=0,
                confidence=eligibility.evidence_confidence, body_region=request.correspondence.body_region,
                start_time_ms=interval[0], end_time_ms=interval[1], evidence=None,
            ),)
        return ComparisonOutcome(analysis=analysis(AnalysisStatus.UNANALYZABLE, eligibility.evidence_confidence,
                                                  findings=findings), reason=reason, **artifacts)
    try:
        alignment = align_dtw(ref.usable_points, learn.usable_points,
                              maximum_cells=policy.maximum_alignment_cells)
    except (AlignmentResourceLimit, NumericalComparisonFailure) as exc:
        reason = OutcomeReason.ALIGNMENT_RESOURCE_LIMIT if isinstance(exc, AlignmentResourceLimit) else OutcomeReason.NUMERICAL_FAILURE
        return ComparisonOutcome(analysis=analysis(AnalysisStatus.FAILED), reason=reason, **artifacts)
    distance = alignment.mean_path_distance
    score = movement_similarity(distance, policy.similarity_distance_scale)
    metrics = ComparisonMetrics(finding_id=request.finding_id, reference=eligibility.reference,
                                learner=eligibility.learner, dtw_cumulative_cost=alignment.cumulative_cost,
                                dtw_mean_path_distance=distance, alignment_path_length=alignment.path_length,
                                movement_similarity=score, evidence_confidence=eligibility.evidence_confidence)
    band = policy.classify(distance)
    finding = StructuredFinding(
        finding_id=request.finding_id, skill=SkillCategory.MOVEMENT, status=band.status, severity=band.severity,
        confidence=eligibility.evidence_confidence, body_region=request.correspondence.body_region,
        start_time_ms=interval[0], end_time_ms=interval[1],
        evidence=Evidence(
            expected_value=f"Reference {request.reference_id}: normalized wrist path; provisional mean DTW distance <= {policy.acceptable_maximum_distance:.6g} shoulder widths",
            observed_value=f"Measured mean DTW distance {distance:.6g} shoulder widths over {alignment.path_length} aligned pairs; engineering similarity {score:.6g}; not UgSL correctness",
            deviation=distance,
        ),
    )
    return ComparisonOutcome(analysis=analysis(AnalysisStatus.COMPLETED, eligibility.evidence_confidence,
                                              score, (finding,)), alignment=alignment, metrics=metrics, **artifacts)
