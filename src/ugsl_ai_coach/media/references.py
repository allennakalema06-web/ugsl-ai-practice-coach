"""Opaque object keys; never paths, URLs, bucket names or endpoints from callers."""

import re

from ugsl_ai_coach.core.config import Settings


class InvalidMediaReference(ValueError):
    pass


class MediaConfigurationError(ValueError):
    pass


def _safe_key(value: str) -> bool:
    return (isinstance(value, str) and len(value) <= 1024
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value) is not None
            and all(part not in ("", ".", "..") for part in value.split("/")))


class MediaReferencePolicy:
    def __init__(self, settings: Settings):
        self.learner_prefix = settings.learner_video_prefix
        self.reference_prefix = settings.reference_profile_prefix
        prefixes = (self.learner_prefix, self.reference_prefix)
        if (any(not p.endswith("/") or not _safe_key(p[:-1]) for p in prefixes)
                or any(a.startswith(b) for a, b in ((prefixes[0], prefixes[1]), (prefixes[1], prefixes[0])))):
            raise MediaConfigurationError("Media namespaces must be valid, distinct and non-overlapping")

    def _validate(self, value: str, prefix: str) -> str:
        if not _safe_key(value) or not value.startswith(prefix) or len(value) <= len(prefix):
            raise InvalidMediaReference("An internal object reference in the required namespace is required")
        return value

    def learner(self, value: str) -> str:
        return self._validate(value, self.learner_prefix)

    def reference(self, value: str) -> str:
        return self._validate(value, self.reference_prefix)
