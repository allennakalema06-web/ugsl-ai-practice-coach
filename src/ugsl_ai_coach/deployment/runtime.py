"""Three production process compositions and one explicit migration operation."""

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.deployment.configuration import api_port, validate_config


def run_role(role: str, settings: Settings):
    settings = validate_config(role, settings)
    from ugsl_ai_coach.core.logging import configure_logging
    configure_logging(settings.log_level)
    from ugsl_ai_coach.infrastructure.postgres.connection import connection_factory
    connect = connection_factory(settings)
    if role == 'migrate':
        from ugsl_ai_coach.infrastructure.postgres.migrate import migrate
        applied = migrate(connect)
        print(f'Migrations complete: {len(applied)} applied')
    elif role == 'api':
        import uvicorn
        from ugsl_ai_coach.main import create_app
        # Passing the existing application avoids import-time worker composition.
        uvicorn.run(create_app(settings), host='0.0.0.0', port=api_port(), workers=1,
                    access_log=False, log_config=None, timeout_graceful_shutdown=60,
                    proxy_headers=False)
    elif role == 'analysis-worker':
        from ugsl_ai_coach.worker.processor import create_production_processor
        from ugsl_ai_coach.worker.runtime import run_postgres_worker
        processor = create_production_processor(settings)
        try:
            # Existing runtime performs only owned-temp cleanup before polling.
            run_postgres_worker(processor, settings)
        finally:
            processor.media.client.close()
    elif role == 'coaching-worker':
        from ugsl_ai_coach.infrastructure.postgres.coaching_work import PostgresCoachingWork
        from ugsl_ai_coach.infrastructure.postgres.telemetry import PostgresWorkerMetrics
        from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
        from ugsl_ai_coach.worker.coaching import CoachingWorker, CoachingProcessor
        from ugsl_ai_coach.worker.runtime import shutdown_signals
        worker = CoachingWorker(PostgresCoachingWork(connect), CoachingProcessor(DeterministicCoachingProvider()),
                                settings, PostgresWorkerMetrics(connect))
        with shutdown_signals(worker):
            worker.run()
