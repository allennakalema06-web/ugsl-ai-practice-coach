-- M6E engineering delivery/capacity state; no learner telemetry or retention policy.
CREATE TABLE coaching_work (
    analysis_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    feedback_id TEXT NOT NULL UNIQUE CHECK (length(btrim(feedback_id)) > 0),
    state TEXT NOT NULL CHECK (state IN ('PENDING','CLAIMED','COMPLETED')),
    delivery_count BIGINT NOT NULL DEFAULT 0 CHECK (delivery_count >= 0),
    failure_count BIGINT NOT NULL DEFAULT 0 CHECK (failure_count >= 0),
    claimed_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    last_error_code TEXT CHECK (last_error_code IN ('COACHING_PROCESSING_ERROR')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (analysis_id, attempt_id) REFERENCES analysis_jobs (analysis_id, attempt_id),
    CHECK ((state = 'CLAIMED' AND delivery_count > 0 AND claimed_at IS NOT NULL
        AND lease_expires_at > claimed_at AND lease_expires_at IS NOT NULL)
        OR (state IN ('PENDING','COMPLETED') AND claimed_at IS NULL AND lease_expires_at IS NULL))
);
CREATE INDEX coaching_work_pending ON coaching_work(next_attempt_at, created_at) WHERE state = 'PENDING';
CREATE INDEX coaching_work_expired ON coaching_work(lease_expires_at) WHERE state = 'CLAIMED';

-- Preserve existing feedback identity; otherwise assign once, before delivery.
INSERT INTO coaching_work (analysis_id, attempt_id, feedback_id, state)
SELECT j.analysis_id, j.attempt_id,
    COALESCE(c.feedback_id, 'FB-' || md5('ugsl-m6e:' || j.analysis_id)::uuid::text),
    CASE WHEN c.analysis_id IS NULL THEN 'PENDING' ELSE 'COMPLETED' END
FROM analysis_jobs j LEFT JOIN coaching_records c USING (analysis_id)
WHERE j.state IN ('COMPLETED','UNANALYZABLE','FAILED')
ON CONFLICT (analysis_id) DO NOTHING;

CREATE FUNCTION schedule_coaching_work() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.state IN ('COMPLETED','UNANALYZABLE','FAILED') THEN
        INSERT INTO coaching_work (analysis_id, attempt_id, feedback_id, state)
        VALUES (NEW.analysis_id, NEW.attempt_id,
            'FB-' || md5('ugsl-m6e:' || NEW.analysis_id)::uuid::text, 'PENDING')
        ON CONFLICT (analysis_id) DO NOTHING;
    END IF;
    RETURN NULL;
END $$;
CREATE TRIGGER schedule_coaching_work AFTER INSERT OR UPDATE ON analysis_jobs
    FOR EACH ROW EXECUTE FUNCTION schedule_coaching_work();

CREATE FUNCTION protect_coaching_work() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Coaching delivery history cannot be deleted' USING ERRCODE = '23514';
    END IF;
    IF OLD.state = 'COMPLETED' OR
       ROW(OLD.analysis_id, OLD.attempt_id, OLD.feedback_id, OLD.created_at) IS DISTINCT FROM
       ROW(NEW.analysis_id, NEW.attempt_id, NEW.feedback_id, NEW.created_at) OR NOT (
        (NEW.state = 'CLAIMED' AND OLD.state IN ('PENDING','CLAIMED')
         AND NEW.delivery_count = OLD.delivery_count + 1 AND NEW.failure_count = OLD.failure_count
         AND ((OLD.state = 'PENDING' AND OLD.next_attempt_at <= clock_timestamp()) OR
              (OLD.state = 'CLAIMED' AND OLD.lease_expires_at <= clock_timestamp()))) OR
        (OLD.state = 'CLAIMED' AND NEW.state = 'PENDING'
         AND OLD.lease_expires_at > clock_timestamp() AND NEW.delivery_count = OLD.delivery_count
         AND NEW.failure_count = OLD.failure_count + 1 AND NEW.next_attempt_at > clock_timestamp()) OR
        (NEW.state = 'COMPLETED' AND NEW.delivery_count = OLD.delivery_count
         AND NEW.failure_count = OLD.failure_count AND EXISTS (
             SELECT 1 FROM coaching_records WHERE analysis_id = OLD.analysis_id))
       ) THEN
        RAISE EXCEPTION 'Invalid coaching delivery transition' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER protect_coaching_work BEFORE UPDATE OR DELETE ON coaching_work
    FOR EACH ROW EXECUTE FUNCTION protect_coaching_work();

-- Also supports the existing append-only M6A repository: record insertion and
-- reconciliation complete in the same transaction, without replacing history.
CREATE FUNCTION reconcile_coaching_work() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE coaching_work SET state = 'COMPLETED', claimed_at = NULL, lease_expires_at = NULL,
        updated_at = clock_timestamp() WHERE analysis_id = NEW.analysis_id AND state <> 'COMPLETED';
    RETURN NULL;
END $$;
CREATE TRIGGER reconcile_coaching_work AFTER INSERT ON coaching_records
    FOR EACH ROW EXECUTE FUNCTION reconcile_coaching_work();

CREATE FUNCTION check_coaching_delivery() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE aid TEXT; terminal BOOLEAN; ws TEXT; has_feedback BOOLEAN;
BEGIN
    aid := NEW.analysis_id;
    SELECT state IN ('COMPLETED','UNANALYZABLE','FAILED') INTO terminal
        FROM analysis_jobs WHERE analysis_id = aid;
    SELECT state INTO ws FROM coaching_work WHERE analysis_id = aid;
    SELECT EXISTS (SELECT 1 FROM coaching_records WHERE analysis_id = aid) INTO has_feedback;
    IF (terminal AND ws IS NULL) OR (ws IS NOT NULL AND NOT terminal) OR
       (ws IS NOT NULL AND ((ws = 'COMPLETED') <> has_feedback)) THEN
        RAISE EXCEPTION 'Inconsistent coaching delivery' USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER check_terminal_coaching AFTER INSERT OR UPDATE ON analysis_jobs
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_coaching_delivery();
CREATE CONSTRAINT TRIGGER check_coaching_work AFTER INSERT OR UPDATE ON coaching_work
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_coaching_delivery();
CREATE CONSTRAINT TRIGGER check_coaching_record AFTER INSERT ON coaching_records
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_coaching_delivery();

-- One rolling fixed-window row per authenticated service/operation, never per request.
CREATE TABLE service_rate_limits (
    principal_digest TEXT NOT NULL CHECK (principal_digest ~ '^[a-f0-9]{64}$'),
    operation TEXT NOT NULL CHECK (operation IN ('submit','read')),
    window_start TIMESTAMPTZ NOT NULL,
    request_count BIGINT NOT NULL CHECK (request_count > 0),
    PRIMARY KEY (principal_digest, operation)
);

-- Bounded technical aggregates only, shared by API and independent worker processes.
CREATE TABLE operational_worker_counts (
    worker TEXT NOT NULL CHECK (worker IN ('analysis','coaching')),
    outcome TEXT NOT NULL CHECK (outcome IN ('success','failure','retry','claim','reclaim')),
    event_count BIGINT NOT NULL CHECK (event_count >= 0),
    PRIMARY KEY (worker, outcome)
);
