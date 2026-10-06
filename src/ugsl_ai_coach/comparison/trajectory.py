"""Selection preserves gaps; no label substitution, mirroring, or interpolation."""

from ugsl_ai_coach.cv.models import ExtractionResult, HandLabel, NormalizationStatus
from ugsl_ai_coach.comparison.models import (
    Availability, InvalidComparisonInput, MovementTrajectory, TrajectoryObservation,
)
from ugsl_ai_coach.comparison.policy import ComparisonPolicy


def select_trajectory(extraction: ExtractionResult, label: HandLabel,
                      policy: ComparisonPolicy) -> MovementTrajectory:
    observations = []
    for frame in extraction.frames:
        matches = [(i, h) for i, h in enumerate(frame.raw.hands) if h.reported_handedness == label]
        state, coordinates, detection_index = Availability.MISSING_HAND, None, None
        if len(matches) > 1:
            state = Availability.AMBIGUOUS_HAND
        elif matches:
            detection_index, hand = matches[0]
            if hand.handedness_confidence is None or hand.handedness_confidence < policy.minimum_handedness_confidence:
                state = Availability.LOW_LABEL_CONFIDENCE
            elif frame.normalization.status != NormalizationStatus.NORMALIZED:
                state = Availability.UNNORMALIZED
            else:
                if len(frame.normalization.hands) != len(frame.raw.hands):
                    raise InvalidComparisonInput("Normalized hand detections do not correspond to raw detections")
                points = frame.normalization.hands[detection_index]
                if len({p.index for p in points}) != len(points):
                    raise InvalidComparisonInput("Normalized landmark indices are ambiguous")
                wrist = next((p for p in points if p.index == 0), None)
                if wrist is None:
                    state = Availability.MISSING_WRIST
                else:
                    state, coordinates = Availability.AVAILABLE, wrist.coordinates
        observations.append(TrajectoryObservation(
            frame_index=frame.frame_index, timestamp_ms=frame.timestamp_ms, availability=state,
            x=coordinates.x if coordinates is not None else None,
            y=coordinates.y if coordinates is not None else None,
            hand_detection_index=detection_index,
        ))
    return MovementTrajectory(reported_label=label, observations=tuple(observations))
