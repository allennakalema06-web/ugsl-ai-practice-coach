"""Versioned minimal M4 reference artifacts, created offline from expert extraction."""

import json
from typing import Literal, Self

from pydantic import Field, StringConstraints, model_validator
from typing import Annotated

from ugsl_ai_coach.comparison.models import ComparisonModel, ExtractionContext, HandCorrespondence, MovementTrajectory
from ugsl_ai_coach.comparison.policy import ComparisonPolicy
from ugsl_ai_coach.comparison.trajectory import select_trajectory
from ugsl_ai_coach.cv.models import ExtractionResult
from ugsl_ai_coach.media.ports import CorruptReferenceProfile, IncompatibleReferenceProfile

Identity = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=1024)]
ReferenceIdentity = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]


class ReferenceProfile(ComparisonModel):
    schema_version: Literal["ugsl-reference-profile-v1"] = "ugsl-reference-profile-v1"
    extraction_version: Literal["m3-shoulder-normalization-v1"] = "m3-shoulder-normalization-v1"
    profile_id: Identity
    reference_id: ReferenceIdentity
    extractor_identity: Identity
    correspondence: HandCorrespondence
    context: ExtractionContext
    policy: ComparisonPolicy
    trajectory: MovementTrajectory

    @model_validator(mode="after")
    def consistent_reference(self) -> Self:
        if self.trajectory.reported_label != self.correspondence.reference_label:
            raise ValueError("Trajectory and established reference label must match")
        if any(p.frame_index >= self.context.video.frame_count or p.timestamp_ms > self.context.video.duration_ms
               for p in self.trajectory.observations):
            raise ValueError("Reference trajectory exceeds its source context")
        return self


def build_reference_profile(extraction: ExtractionResult, *, profile_id: str, reference_id: str,
                            extractor_identity: str, correspondence: HandCorrespondence,
                            policy: ComparisonPolicy | None = None) -> ReferenceProfile:
    extraction = ExtractionResult.model_validate(extraction.model_dump(mode="python"))
    policy = policy if policy is not None else ComparisonPolicy()
    return ReferenceProfile(profile_id=profile_id, reference_id=reference_id, extractor_identity=extractor_identity,
        correspondence=correspondence, context=ExtractionContext.from_extraction(extraction), policy=policy,
        trajectory=select_trajectory(extraction, correspondence.reference_label, policy))


def serialize_profile(profile: ReferenceProfile) -> bytes:
    checked = ReferenceProfile.model_validate(profile.model_dump(mode="python"))
    return checked.model_dump_json().encode("utf-8")


def load_profile(data: bytes, *, profile_id: str, extractor_identity: str, policy: ComparisonPolicy) -> ReferenceProfile:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
        if not isinstance(value, dict):
            raise ValueError
        if "schema_version" not in value:
            raise ValueError("Missing schema version")
        if value["schema_version"] != "ugsl-reference-profile-v1":
            raise IncompatibleReferenceProfile("Reference schema version is incompatible")
        if "extraction_version" not in value:
            raise ValueError("Missing extraction version")
        if value["extraction_version"] != "m3-shoulder-normalization-v1":
            raise IncompatibleReferenceProfile("Reference schema or extraction version is incompatible")
        profile = ReferenceProfile.model_validate(value)
    except IncompatibleReferenceProfile:
        raise
    except (ValueError, TypeError, RecursionError):
        raise CorruptReferenceProfile("Reference profile violates the typed contract") from None
    if profile.profile_id != profile_id or profile.extractor_identity != extractor_identity or profile.policy != policy:
        raise IncompatibleReferenceProfile("Reference identity, models or comparison policy are incompatible")
    return profile
