"""Bounded cross-process worker aggregates; best-effort, never evidence writes."""

class PostgresWorkerMetrics:
    def __init__(self, connect):
        self.connect = connect

    def outcome(self, worker, outcome):
        if worker not in ('analysis', 'coaching') or outcome not in ('success', 'failure', 'retry', 'claim', 'reclaim'):
            raise ValueError('Unbounded operational labels')
        with self.connect() as conn:
            conn.execute("""INSERT INTO operational_worker_counts (worker, outcome, event_count)
                VALUES (%s, %s, 1) ON CONFLICT (worker, outcome) DO UPDATE
                SET event_count = operational_worker_counts.event_count + 1""", (worker, outcome))

    def claim(self, worker, deliveries):
        self.outcome(worker, 'reclaim' if deliveries > 1 else 'claim')
