# M6F deployment and staging verification

**Deployment target: Render.** This is a new M6F engineering decision, not a
provider prescribed by the UgSL source documents. Render supports the required
Docker web service, separate background workers, managed PostgreSQL, pre-deploy
migrations, health checks and secret configuration. Private S3-compatible object
storage remains external and provider-neutral.

This package is prepared for review. Nothing has been deployed. Live deployment
and end-to-end verification remain the final release gate. Commands below are
operator actions after review, not automatic setup scripts.

## Topology and first-sync decisions

The UgSL platform backend calls the API over HTTPS with service bearer auth.
The API persists jobs in PostgreSQL; the analysis worker retrieves exact pinned
private-object versions and runs M3 -> M4 -> immutable M2. PostgreSQL schedules
separate durable coaching work; the coaching worker runs grounded deterministic
M5 and persists a CoachingRecord. API feedback reads that historical record.
There are exactly three application processes, one managed database, and no broker.

All four Blueprint resources explicitly use **Frankfurt**. Render currently has
no Africa region; this initial region is an engineering choice requiring latency
and residency review before first sync. Region cannot be changed in place after
creation: plan replacement/migration rather than casually editing the field.
See [Render regions](https://render.com/docs/regions) and the
[Blueprint specification](https://render.com/docs/blueprint-spec).

Each application initially uses paid `0.5c-512mb`, one instance; PostgreSQL uses
paid `0.1c-256mb`, major version 18. These are validation sizes, not proven
production capacity. Autoscaling is absent. Observe analysis-worker memory during
actual MediaPipe execution. If it approaches 512 MB or OOMs, increase that worker
to `1c-2g`; do not weaken the pipeline to fit. Observe CPU, queue latency, database
connections and storage as well. All services have `autoDeployTrigger: off`.
A Git push does not authorize a production deployment.

`UGSL_DATABASE_URL` uses `fromDatabase.connectionString`, the internal URL.
`ipAllowList: []` blocks external database access; internal services in the same
region still connect. Review the provider's actual network/access settings during
sync. Do not enable public DB access merely to run a local harness: it uses HTTPS
and S3, not SQL. Database backup/PITR is an infrastructure capability: verify the
chosen paid plan's retention, restore procedure and recovery objectives in Render
before production approval. Perform a deliberate restore rehearsal to an isolated
database. Free PostgreSQL is not a production choice. Python implements no backups.

## Runtime commands and configuration

Use the same installed image with one command per container:

```sh
python -m ugsl_ai_coach.deployment check-config api
python -m ugsl_ai_coach.deployment check-config analysis-worker
python -m ugsl_ai_coach.deployment check-config coaching-worker
python -m ugsl_ai_coach.deployment check-config migrate
python -m ugsl_ai_coach.deployment migrate
python -m ugsl_ai_coach.deployment api
python -m ugsl_ai_coach.deployment analysis-worker
python -m ugsl_ai_coach.deployment coaching-worker
```

`check-config` validates locally without connecting to PostgreSQL/S3. Runtime
credential usability remains a live check. Failure messages contain no values or
provider exception text. No module import or process startup migrates implicitly.
Only `migrate` invokes the existing ordered 001/002 runner, with transaction,
advisory lock and checksum tracking. No migration 003 is needed. All three services
use that command in pre-deploy, so concurrent/repeated deploys safely serialize.

| Role | Required configuration |
| --- | --- |
| API | Database, strong random service token, private object-store bucket/config; validated rate-limit defaults; optional `PORT` (8000 default) |
| Analysis | Database, bucket/config and usable Boto credential chain, local model assets, validated worker/temp settings |
| Coaching | Database and validated coaching/retry settings; deterministic M5, no service token or AWS credentials |
| Migration | Database only |

Set `UGSL_ENVIRONMENT=production`. API docs/OpenAPI URLs default off in production;
`UGSL_API_DOCS_ENABLED` can explicitly override this outside the reviewed Blueprint.
`GET /api/v1/contracts/analysis` remains available. Service auth and scopes remain
unchanged, with learner authentication/authorization owned by the platform backend.
There is no CORS/frontend integration. API uses one Uvicorn process, `0.0.0.0:$PORT`,
no raw access log, no proxy-header trust. Safe M6E request logging stays enabled.

The Blueprint's `sync: false` fields prompt for secrets/configuration. Generate the
service token with a cryptographic generator (at least 32 random bytes); never use
examples as credentials. API receives DB/token/storage; analysis DB/storage;
coaching DB only. Do not share environment groups that reintroduce extra credentials.
The existing Boto credential chain is used. `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` can be supplied via Render; session credentials also require
an explicitly managed `AWS_SESSION_TOKEN` and renewal plan. No credential is baked
into an image/YAML. For native AWS S3, omit or leave the endpoint field empty;
for another compatible provider set `UGSL_OBJECT_STORE_ENDPOINT_URL` to HTTPS.
Production rejects configured HTTP endpoints and URL-embedded credentials.

## Private object-store checklist

- Operator creates/configures a private bucket; no public-read policy or public ACL.
- Limit API credentials to required bucket/prefix metadata/read operations; analysis
  credentials to metadata/read of those prefixes. The API pins references before
  acceptance; both roles need the adapter's metadata capabilities.
- Separate operator E2E upload credentials permit PutObject only in isolated
  `learner-videos/e2e-staging/` and `reference-profiles/e2e-staging/` prefixes,
  plus the read/metadata permissions needed by the operator tool.
- Versioning is strongly preferred. Verify actual compatible-provider version/ETag
  semantics, MIME types, encryption and access restrictions using staging objects.
- Lifecycle rules must preserve pinned historical versions and profile traceability.
  There is no final learner-media retention/deletion policy introduced here.
- Do not log keys, signed URLs, credentials or media contents. Never make credentials
  or direct object access part of the learner-facing API.

## Image, model assets and CI

The Python 3.13 slim image installs production dependencies and minimal native
GL/GLib/PortAudio libraries for OpenCV/MediaPipe imports. UID/GID 10001 runs the
process; `/var/tmp/ugsl` is writable with mode 0700. Exec-form CMD and Render's
command override yield one process, with no supervisor. Docker COPY is an allowlist;
`.dockerignore` additionally excludes secrets, VCS, local models, media and test output.

The exact existing M3 hand/pose-lite version-1 bytes are packaged with hashes,
source URLs, model-card licensing and Apache 2.0 notice/license. `model_paths()`
checks both hashes. Wheel-installed paths remain stable and M6D extractor identity
uses the same actual model hashes. No startup downloads or model replacements.
Custom model overrides require compatible expert profiles. Bundling does not claim
that MediaPipe or synthetic fixtures establish UgSL linguistic accuracy.

GitHub Actions runs on PR and push to main with `contents: read`, Python 3.13 and
disposable PostgreSQL 18. It runs the complete suite, then a JUnit gate requiring
all 121 PostgreSQL cases with no skips/failures, `pip check` and `git diff --check`.
It builds/installs a wheel outside the checkout, verifies models/native imports,
builds Docker and runs credential-free native-import/CLI-help smokes. No deployment
or registry push occurs. Docker and Render CLI are unavailable on the preparation
machine; Linux image build and provider Blueprint validation remain pending CI/sync.

## Deliberate deployment sequence

1. Review region/residency, plans, private storage, permissions and backup/restore.
2. Review and commit/push separately; confirm GitHub CI is green, including real PG
   execution, installed-wheel verification, Linux image build and image smokes.
3. Operator creates the Render Blueprint from the reviewed commit. Review the exact
   resource list/cost/region before first sync; enter the `sync: false` configuration.
4. Confirm PostgreSQL creation, internal URL wiring, external-access restrictions
   and backup/PITR capability. Confirm private storage credentials per role.
5. Deliberately deploy API and both workers. Inspect successful pre-deploy 001/002
   migrations and clean process startup. Record artifact/commit and configuration
   version without recording secret values.
6. Check liveness, readiness, authenticated metrics, staging E2E, worker recovery,
   log privacy, memory and queue latency using the checks below.
7. Upgrade analysis compute if needed, rerun gates; approve production only after
   live evidence is reviewed. Automatic deployments remain off.

Render health checks use `/api/v1/health` liveness. `/api/v1/health/ready` separately
reports 503 for unavailable PostgreSQL without forcing restart loops. Readiness
does not prove model/storage correctness; E2E does. Shutdown windows are API 60s,
each worker 300s. Existing signals stop polling and allow work to reach a safe
boundary when possible; expired DB-clock leases and generation fences recover
interrupted work. Never delete work to recover a deploy.

## Operator-only staging E2E

Normal pytest never calls external APIs or uploads media. The tool requires all of:

```text
UGSL_E2E_ENVIRONMENT=staging
UGSL_E2E_BASE_URL=https://<reviewed-staging-host>
UGSL_E2E_ALLOWED_HOST=<the-exact-same-host>
UGSL_E2E_SERVICE_TOKEN=<operator-supplied-staging-secret>
UGSL_OBJECT_STORE_BUCKET=<private-staging-bucket>
UGSL_OBJECT_STORE_REGION=<reviewed-region>
UGSL_OBJECT_STORE_ENDPOINT_URL=<HTTPS-compatible-endpoint-or-absent-for-AWS>
AWS credential chain with isolated staging upload permissions
```

Load secrets through the operator's secret manager/environment, never shell history
or committed files. The hostname gate rejects production-labelled hosts, but cannot
prove infrastructure identity: the operator must confirm that the allowlisted host,
bucket and credentials are staging, isolated from learner data. Use local model
overrides only when they match the deployed worker's model hashes.

```sh
python -m ugsl_ai_coach.deployment.e2e --acknowledge-staging-uploads
```

The default creates a short black MJPEG AVI, performs real local M3 extraction and
uploads its valid versioned precomputed reference profile. The deployed worker
runs the real processor; absent hand evidence should honestly produce
**UNANALYZABLE**, then validated persisted deterministic M5 feedback. It is an
infrastructure abstention smoke, not successful signer assessment. For separately
reviewed non-sensitive test media and its matching versioned expert profile:

```sh
python -m ugsl_ai_coach.deployment.e2e --acknowledge-staging-uploads \
  --learner-video /operator/test.avi --reference-profile /operator/profile.json \
  --expected-status COMPLETED
```

Declare the real expected terminal status; the harness never forces COMPLETED.
Original and rewritten profiles pass the production loader/model/policy checks.
Each run uses a cryptographic random ID inside isolated allowed E2E prefixes. No
username/email/PII is used; output contains only check summaries and exercised path.
HTTP redirects are refused to protect credentials. Local temporary media is removed;
remote objects remain for reviewed traceability/retention handling, with no automatic
history deletion or invented lifecycle policy.

The tool checks 200 liveness/readiness, unauthenticated submission 401, authorized
202, stable idempotent identity, bounded progress polling, validated terminal M2,
feedback PENDING/200 with grounded M5 and matching analysis/attempt, stable historical
feedback reread, response reference privacy and authenticated healthy metrics.
`/metrics` requires `metrics:read`. The current static service token grants all
service scopes, so it cannot supply a valid restricted-scope credential. A negative
403 live scope probe is **not configured** in this default topology; existing tests
verify the scope boundary. If an operator already has a reviewed scoped authenticator,
`UGSL_E2E_NO_METRICS_SCOPE_TOKEN` enables the negative probe. Do not invent a second
token or claim that live scope denial passed when only 401 was checked.

## Durable recovery checks (live staging gates)

Use Render's operator stop/suspend controls, never test-only HTTP control routes.

1. Suspend the analysis worker; submit an isolated staging job through the public
   contract. Confirm accepted identity remains readable/nonterminal and
   `ugsl_queue_depth{worker="analysis"}` grows. Restart the worker; poll the same
   identity to its honest terminal M2 and persisted feedback. Do not resubmit a new
   job to hide loss. Also interrupt one active staging job, allow its DB lease to
   expire, restart and verify recovery with no duplicate terminal result.
2. Suspend coaching only; let analysis reach immutable terminal M2. Feedback must
   remain 202 PENDING and `ugsl_queue_depth{worker="coaching"}` grow. Restart coaching;
   feedback becomes 200 for the same analysis/attempt and remains identical on
   reread. Queue depth returns toward zero. M2 payload before/after must be identical.
3. Review worker outcome/reclaim aggregates, readiness and PostgreSQL health. Record
   outcomes privately; do not paste payloads/keys/tokens into operational logs.

## Metrics, alert delivery and log privacy (live gates)

Configure a separately operated Prometheus-compatible collector using
`ops/prometheus/prometheus.example.yml`: HTTPS, bearer credentials mounted as a
restricted file, existing `alerts.yml` mounted at the example rule path. Update the
placeholder host, validate configuration and inspect scrape success. No collector
is added to the Blueprint. Alertmanager/delivery-provider selection and routing are
unresolved: connect the rules deliberately and test delivery before claiming alerts
are operational. No Slack/PagerDuty/email credentials are included.

Verify missing bearer credentials return 401 and authenticated `/metrics` returns
200 with `ugsl_postgres_up 1.0`. Verify queue changes during suspension/restart,
coaching queue recovery, bounded labels and absence of tokens/analysis/attempt/media
identifiers in labels. Review collector access and token rotation procedures.

Inspect representative Render API/worker logs for submission, terminal analysis,
coaching completion, auth rejection and an isolated safely induced rate-limit event.
Expect operational event/request IDs; exclude Authorization/token, object keys,
signed URLs, DB URLs, AWS secrets, raw findings/feedback/request bodies and learner
PII. Verify failure-path logs too. Application access logging is controlled by M6E;
Render edge/platform logging is provider infrastructure and needs separate review.
This check has **not** passed until actual deployed logs have been inspected.

## Rollback and remaining release gates

Rollback application artifacts deliberately, keep the database and immutable history.
001 establishes the existing persistence contract; 002 adds durable coaching,
bounded rate limiting and telemetry without replacing M2/coaching records. Its
reconciliation trigger supports the earlier append-only coaching writer. M6A/M6B
contract code remains schema-compatible, but those milestones are not complete
three-process deployments. M6C/M6D packages containing only migration 001 have an
unknown-migration guard: their old migration command rejects a DB already at 002.
Use a reviewed rollback artifact retaining the unchanged 001/002 migration files
and compatible runner (the current M6E baseline is the intended rollback target).
Do not bypass checksums, remove tracking rows, drop tables or invent down-migrations.
Check worker/record compatibility before restart and rerun live checks. Hypothetical
future migrations require their own compatibility and backup/restore plan; universal
rollback safety is not claimed.

Still pending: reviewed push and green Linux/PG18 CI; provider Blueprint validation;
private storage and access/versioning review; deliberate resource creation/deploy;
pre-deploy migrations and three-process startup; liveness/readiness/metrics; real
staging E2E and honest result path; both recovery exercises; actual log/edge privacy;
memory/capacity observation; backup/PITR restore rehearsal; collector integration
and unresolved alert delivery. No external resource creation/upload/deploy is
authorized by this preparation task.
