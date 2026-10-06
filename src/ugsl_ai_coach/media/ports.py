from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol


class MediaError(RuntimeError):
    """Safe media failure; never a learner-analysis outcome."""


class LearnerMediaNotFound(MediaError): pass
class ReferenceProfileNotFound(MediaError): pass
class ObjectStoreUnavailable(MediaError): pass
class UnsupportedLearnerVideo(MediaError): pass
class MediaTooLarge(MediaError): pass
class CorruptReferenceProfile(MediaError): pass
class IncompatibleReferenceProfile(MediaError): pass


class MediaStore(Protocol):
    def learner_video(self, reference: str) -> AbstractContextManager[Path]: ...
    def reference_profile(self, reference: str) -> bytes: ...


class MediaReferenceResolver(Protocol):
    def pin_learner(self, reference: str) -> str: ...
    def pin_reference(self, reference: str) -> str: ...
