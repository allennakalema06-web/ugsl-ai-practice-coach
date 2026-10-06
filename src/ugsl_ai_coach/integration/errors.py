"""Integration errors have no HTTP status or learner-performance meaning."""


class IntegrationError(ValueError):
    """Invalid integration operation; messages contain no media references."""


class IdempotencyConflict(IntegrationError):
    """A key was already accepted for a different logical payload."""


class JobNotFound(IntegrationError):
    """No job exists for the requested analysis identifier."""


class InvalidTransition(IntegrationError):
    """A lifecycle update would skip, repeat or rewrite historical state."""


class ConcurrentUpdate(IntegrationError):
    """The stored snapshot changed before an atomic update could apply."""


class AnalysisIdConflict(IntegrationError):
    """A newly allocated analysis identifier already identifies another job."""


class AdapterContractError(IntegrationError):
    """An injected adapter returned a record inconsistent with the operation."""


class AnalysisNotTerminal(IntegrationError):
    """Coaching persistence requires existing terminal analysis evidence."""


class FeedbackConflict(IntegrationError):
    """An analysis already has a different immutable coaching artifact."""
