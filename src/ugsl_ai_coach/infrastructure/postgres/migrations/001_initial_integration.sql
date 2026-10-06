CREATE TABLE analysis_jobs (
    analysis_id TEXT PRIMARY KEY CHECK (analysis_id ~ '^AN-[0-9]{6,}$'),
    attempt_id TEXT NOT NULL CHECK (attempt_id ~ '^ATT-[0-9]{6,}$'),
    idempotency_key TEXT NOT NULL UNIQUE CHECK (length(btrim(idempotency_key)) > 0),
    learner_video_ref TEXT NOT NULL CHECK (length(btrim(learner_video_ref)) > 0),
    reference_profile_ref TEXT NOT NULL CHECK (length(btrim(reference_profile_ref)) > 0),
    state TEXT NOT NULL CHECK (state IN ('SUBMITTED','PROCESSING','COMPLETED','UNANALYZABLE','FAILED')),
    structured_analysis JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (analysis_id, attempt_id, learner_video_ref, reference_profile_ref),
    UNIQUE (analysis_id, attempt_id),
    CHECK ((state IN ('SUBMITTED','PROCESSING') AND structured_analysis IS NULL) OR
           (state IN ('COMPLETED','UNANALYZABLE','FAILED') AND structured_analysis IS NOT NULL
            AND jsonb_typeof(structured_analysis) = 'object'
            AND structured_analysis->>'analysis_id' IS NOT DISTINCT FROM analysis_id
            AND structured_analysis->>'attempt_id' IS NOT DISTINCT FROM attempt_id
            AND structured_analysis->>'status' IS NOT DISTINCT FROM state))
);

CREATE TABLE analysis_work (
    analysis_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    learner_video_ref TEXT NOT NULL,
    reference_profile_ref TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('PENDING','CLAIMED','COMPLETED')),
    delivery_count BIGINT NOT NULL DEFAULT 0 CHECK (delivery_count >= 0),
    claimed_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (analysis_id, attempt_id, learner_video_ref, reference_profile_ref)
        REFERENCES analysis_jobs (analysis_id, attempt_id, learner_video_ref, reference_profile_ref),
    CHECK ((state = 'CLAIMED' AND delivery_count > 0 AND claimed_at IS NOT NULL
            AND lease_expires_at IS NOT NULL AND lease_expires_at > claimed_at
            AND claimed_at >= TIMESTAMPTZ '1970-01-01 00:00:00+00'
            AND claimed_at = date_trunc('milliseconds', claimed_at)
            AND lease_expires_at = date_trunc('milliseconds', lease_expires_at)) OR
           (state IN ('PENDING','COMPLETED') AND claimed_at IS NULL AND lease_expires_at IS NULL
            AND (state <> 'COMPLETED' OR delivery_count > 0)))
);
CREATE INDEX analysis_work_pending ON analysis_work (created_at, analysis_id) WHERE state = 'PENDING';
CREATE INDEX analysis_work_expired ON analysis_work (lease_expires_at, created_at) WHERE state = 'CLAIMED';

CREATE TABLE coaching_records (
    analysis_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    feedback_id TEXT NOT NULL UNIQUE CHECK (length(btrim(feedback_id)) > 0),
    feedback JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (analysis_id, attempt_id) REFERENCES analysis_jobs (analysis_id, attempt_id),
    CHECK (jsonb_typeof(feedback) = 'object'
           AND feedback->>'analysis_id' IS NOT DISTINCT FROM analysis_id
           AND feedback->>'attempt_id' IS NOT DISTINCT FROM attempt_id
           AND feedback->>'feedback_id' IS NOT DISTINCT FROM feedback_id)
);

-- Defense in depth for writes outside the narrow adapter operations.
CREATE FUNCTION protect_analysis_history() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Analysis history cannot be deleted' USING ERRCODE = '23514';
    END IF;
    IF OLD.state IN ('COMPLETED','UNANALYZABLE','FAILED') OR
       ROW(OLD.analysis_id, OLD.attempt_id, OLD.idempotency_key, OLD.learner_video_ref,
           OLD.reference_profile_ref, OLD.created_at) IS DISTINCT FROM
       ROW(NEW.analysis_id, NEW.attempt_id, NEW.idempotency_key, NEW.learner_video_ref,
           NEW.reference_profile_ref, NEW.created_at) OR
       NOT ((OLD.state = 'SUBMITTED' AND NEW.state = 'PROCESSING') OR
            (OLD.state = 'PROCESSING' AND NEW.state IN ('COMPLETED','UNANALYZABLE','FAILED'))) THEN
        RAISE EXCEPTION 'Invalid immutable analysis transition' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER protect_analysis_history BEFORE UPDATE OR DELETE ON analysis_jobs
    FOR EACH ROW EXECUTE FUNCTION protect_analysis_history();

CREATE FUNCTION protect_work_history() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Work history cannot be deleted' USING ERRCODE = '23514';
    END IF;
    IF OLD.state = 'COMPLETED' OR
       ROW(OLD.analysis_id, OLD.attempt_id, OLD.learner_video_ref, OLD.reference_profile_ref, OLD.created_at)
       IS DISTINCT FROM
       ROW(NEW.analysis_id, NEW.attempt_id, NEW.learner_video_ref, NEW.reference_profile_ref, NEW.created_at) OR
       NOT ((OLD.state IN ('PENDING','CLAIMED') AND NEW.state = 'CLAIMED'
             AND NEW.delivery_count = OLD.delivery_count + 1
             AND (OLD.state = 'PENDING' OR OLD.lease_expires_at <= clock_timestamp())) OR
            (OLD.state = 'CLAIMED' AND NEW.state = 'COMPLETED'
             AND NEW.delivery_count = OLD.delivery_count
             AND OLD.claimed_at <= clock_timestamp() AND OLD.lease_expires_at > clock_timestamp())) THEN
        RAISE EXCEPTION 'Invalid immutable work transition' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER protect_work_history BEFORE UPDATE OR DELETE ON analysis_work
    FOR EACH ROW EXECUTE FUNCTION protect_work_history();

-- Deferred checks permit two inserts/updates in a transaction, never a half-commit.
CREATE FUNCTION check_job_work_pair() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE job_state TEXT; work_state TEXT; deliveries BIGINT;
BEGIN
    SELECT j.state, w.state, w.delivery_count INTO job_state, work_state, deliveries
        FROM analysis_jobs j LEFT JOIN analysis_work w USING (analysis_id)
        WHERE j.analysis_id = NEW.analysis_id;
    IF job_state IS NULL OR work_state IS NULL OR
       ((job_state IN ('COMPLETED','UNANALYZABLE','FAILED')) <> (work_state = 'COMPLETED')) OR
       (job_state = 'PROCESSING' AND deliveries = 0) THEN
        RAISE EXCEPTION 'Inconsistent job/work pair' USING ERRCODE = '23514';
    END IF;
    RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER check_job_pair AFTER INSERT OR UPDATE ON analysis_jobs
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_job_work_pair();
CREATE CONSTRAINT TRIGGER check_work_pair AFTER INSERT OR UPDATE ON analysis_work
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION check_job_work_pair();

CREATE FUNCTION protect_coaching_history() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP <> 'INSERT' THEN
        RAISE EXCEPTION 'Coaching history is append-only' USING ERRCODE = '23514';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM analysis_jobs WHERE analysis_id = NEW.analysis_id
                   AND attempt_id = NEW.attempt_id AND state IN ('COMPLETED','UNANALYZABLE','FAILED')
                   AND state IS NOT DISTINCT FROM NEW.feedback->>'overall_status') THEN
        RAISE EXCEPTION 'Coaching requires matching terminal analysis' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER protect_coaching_history BEFORE INSERT OR UPDATE OR DELETE ON coaching_records
    FOR EACH ROW EXECUTE FUNCTION protect_coaching_history();
