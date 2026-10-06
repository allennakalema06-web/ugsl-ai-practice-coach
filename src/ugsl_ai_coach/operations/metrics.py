"""Process counters plus restart-safe DB queue gauges, with bounded labels."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.http = Counter('ugsl_http_requests_total', 'HTTP requests', ['route', 'method', 'status'], registry=self.registry)
        self.latency = Histogram('ugsl_http_request_duration_seconds', 'HTTP duration', ['route', 'method'], registry=self.registry)
        self.rate = Counter('ugsl_rate_limit_rejections_total', 'Rejected service requests', ['operation'], registry=self.registry)
        self.work = Counter('ugsl_worker_outcomes_total', 'Worker outcomes', ['worker', 'outcome'], registry=self.registry)
        self.claims = Counter('ugsl_work_claims_total', 'Claims and reclaims', ['worker', 'kind'], registry=self.registry)
        self.terminals = Counter('ugsl_analysis_terminal_total', 'Authentic terminal results', ['status'], registry=self.registry)
        self.postgres = Gauge('ugsl_postgres_up', 'DB snapshot availability', registry=self.registry)
        self.depth = Gauge('ugsl_queue_depth', 'Uncompleted durable work', ['worker'], registry=self.registry)
        self.age = Gauge('ugsl_oldest_eligible_work_age_seconds', 'Age of oldest eligible work', ['worker'], registry=self.registry)
        self.shared = Gauge('ugsl_durable_worker_events_total', 'Best-effort cross-process operational aggregates',
                            ['worker', 'outcome'], registry=self.registry)
        self.completed = Gauge('ugsl_coaching_completed', 'Committed coaching delivery count', registry=self.registry)
        self.results = Gauge('ugsl_analysis_results', 'Immutable terminal analysis counts', ['status'], registry=self.registry)

    def record_http(self, route, method, status, duration):
        # Values come from route templates, normalized methods/status, never paths.
        self.http.labels(route, method, str(status)).inc()
        self.latency.labels(route, method).observe(duration)

    def outcome(self, worker, outcome):
        if worker not in ('analysis', 'coaching') or outcome not in ('success', 'failure', 'retry'):
            raise ValueError('Unbounded work labels')
        self.work.labels(worker, outcome).inc()

    def claim(self, worker, deliveries):
        self.claims.labels(worker, 'reclaim' if deliveries > 1 else 'claim').inc()

    def snapshot(self, connect):
        try:
            with connect() as conn:
                conn.execute("SELECT 1")
                for row in conn.execute('SELECT worker, outcome, event_count FROM operational_worker_counts').fetchall():
                    if row['worker'] in ('analysis', 'coaching') and row['outcome'] in ('success', 'failure', 'retry', 'claim', 'reclaim'):
                        self.shared.labels(row['worker'], row['outcome']).set(row['event_count'])
                self.completed.set(conn.execute("SELECT count(*) AS n FROM coaching_work WHERE state = 'COMPLETED'").fetchone()['n'])
                counts = {row['state']: row['n'] for row in conn.execute("""SELECT state, count(*) AS n FROM analysis_jobs
                    WHERE state IN ('COMPLETED','UNANALYZABLE','FAILED') GROUP BY state""").fetchall()}
                for state in ('COMPLETED', 'UNANALYZABLE', 'FAILED'):
                    self.results.labels(state).set(counts.get(state, 0))
                for worker, query in (
                    ('analysis', """SELECT count(*) FILTER (WHERE state <> 'COMPLETED') AS depth,
                        COALESCE(EXTRACT(epoch FROM clock_timestamp() - min(created_at) FILTER (
                        WHERE state = 'PENDING' OR (state = 'CLAIMED' AND lease_expires_at <= clock_timestamp()))), 0) AS age
                        FROM analysis_work"""),
                    ('coaching', """SELECT count(*) FILTER (WHERE state <> 'COMPLETED') AS depth,
                        COALESCE(EXTRACT(epoch FROM clock_timestamp() - min(created_at) FILTER (
                        WHERE (state = 'PENDING' AND next_attempt_at <= clock_timestamp()) OR
                        (state = 'CLAIMED' AND lease_expires_at <= clock_timestamp()))), 0) AS age FROM coaching_work""")):
                    row = conn.execute(query).fetchone()
                    self.depth.labels(worker).set(row['depth'])
                    self.age.labels(worker).set(max(0, float(row['age'])))
            self.postgres.set(1)
        except Exception:
            from ugsl_ai_coach.operations.events import emit
            emit('database_unavailable', error_code='METRICS_SNAPSHOT_UNAVAILABLE')
            self.postgres.set(0)
            # Do not publish stale queue truth after a failed snapshot.
            self.depth.clear()
            self.age.clear()
            self.shared.clear()
            self.completed.set(float('nan'))
            self.results.clear()

    def render(self):
        return generate_latest(self.registry)


worker_metrics = Metrics()  # No connections or network at import.


def best_effort(callback, *args):
    try:
        callback(*args)
    except Exception:
        pass
