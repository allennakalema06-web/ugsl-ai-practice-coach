"""Deterministically authorize explanations from validated M2 findings only."""

from ugsl_ai_coach.coaching import policy
from ugsl_ai_coach.coaching.models import (
    ActionKind, AuthorizedFact, CoachingContext, CoachingPoint,
)
from ugsl_ai_coach.domain.analysis import (
    AnalysisStatus, FindingStatus, SkillCategory, StructuredAnalysisResult,
)


def ground_analysis(analysis: StructuredAnalysisResult, feedback_id: str) -> CoachingContext:
    if not isinstance(analysis, StructuredAnalysisResult):
        raise TypeError("Coaching requires a StructuredAnalysisResult")
    # Revalidate even when callers used unchecked Pydantic copy/construction APIs.
    analysis = StructuredAnalysisResult.model_validate(analysis.model_dump(mode="python"))
    ids = [finding.finding_id for finding in analysis.findings]
    if len(ids) != len(set(ids)):
        raise ValueError("Finding identifiers must be unique for coaching traceability")
    categories: dict[str, list[AuthorizedFact]] = {
        "strengths": [], "corrections": [], "observations": [],
    }
    for finding in analysis.findings:
        if finding.skill != SkillCategory.MOVEMENT:
            continue
        category = (
            "strengths" if finding.status == FindingStatus.STRONG else
            "corrections" if finding.status == FindingStatus.NEEDS_IMPROVEMENT else
            "observations"
        )
        categories[category].append(AuthorizedFact(
            point=CoachingPoint(
                finding_id=finding.finding_id, skill=finding.skill,
                body_region=finding.body_region,
                start_time_ms=finding.start_time_ms, end_time_ms=finding.end_time_ms,
                message=policy.message_for(finding.status, limited_evidence=(
                    finding.confidence < 1 or (
                        analysis.overall_confidence is not None and analysis.overall_confidence < 1
                    )
                )),
                reason=policy.REASONS[finding.status],
            ),
            source_status=finding.status, severity=finding.severity,
            confidence=finding.confidence,
        ))
    # Prioritize only existing evidence. Stable identifiers resolve exact ties.
    selected = {
        category: tuple(sorted(facts, key=lambda fact: (
            -fact.severity, -fact.confidence, fact.point.finding_id,
        ))[:policy.POLICY.maximum_points_per_category])
        for category, facts in categories.items()
    }
    if analysis.status == AnalysisStatus.FAILED:
        summary, action = policy.FAILED_SUMMARY, ActionKind.TRY_LATER
    elif analysis.status == AnalysisStatus.UNANALYZABLE:
        summary, action = policy.UNANALYZABLE_SUMMARY, ActionKind.RETRY_CAPTURE
    else:
        summary = policy.COMPLETED_SUMMARY if any(selected.values()) else policy.EMPTY_SUMMARY
        if any(f.skill == SkillCategory.MOVEMENT and (
            f.confidence < 1 or analysis.overall_confidence < 1 or
            f.status in (FindingStatus.WARNING, FindingStatus.INSUFFICIENT_EVIDENCE)
        ) for f in analysis.findings):
            summary = policy.UNCERTAIN_SUMMARY
        if selected["corrections"]:
            action = ActionKind.REVIEW_MOVEMENT
        elif any(f.source_status in (FindingStatus.INSUFFICIENT_EVIDENCE, FindingStatus.WARNING)
                 for f in selected["observations"]):
            action = ActionKind.RETRY_CAPTURE
        else:
            action = ActionKind.RETRY_ATTEMPT
    return CoachingContext(
        feedback_id=feedback_id, attempt_id=analysis.attempt_id,
        analysis_id=analysis.analysis_id, overall_status=analysis.status,
        evidence_confidence=analysis.overall_confidence,
        **selected, summary=summary, recommended_action=policy.ACTIONS[action],
        encouragement=policy.ENCOURAGEMENT,
        policy_version=policy.POLICY.policy_version,
        template_version=policy.POLICY.template_version,
        constraints=policy.POLICY.constraints,
    )
