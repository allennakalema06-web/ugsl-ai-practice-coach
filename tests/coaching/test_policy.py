from ugsl_ai_coach.coaching.policy import CONSTRAINTS, POLICY


def test_central_constraints_cover_future_provider_boundary():
    assert POLICY.policy_version == "bai-coaching-v1"
    assert POLICY.template_version == "movement-coaching-v1"
    assert POLICY.maximum_points_per_category == 1
    assert POLICY.constraints == CONSTRAINTS
    rules = " ".join(CONSTRAINTS).lower()
    for topic in ("evidence", "ugsl", "severity", "timestamps", "direction", "handshape",
                  "orientation", "location", "timing", "grades", "uncertainty", "retry",
                  "agency", "shame", "urgency", "audio", "reject"):
        assert topic in rules
