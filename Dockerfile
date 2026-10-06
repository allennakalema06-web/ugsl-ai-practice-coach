FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UGSL_ENVIRONMENT=production \
    UGSL_API_DOCS_ENABLED=false \
    TMPDIR=/var/tmp/ugsl

# OpenCV GL/GLib, MediaPipe EGL runtime and sounddevice import; no desktop stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libegl1 libglib2.0-0 libportaudio2 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 ugsl \
    && useradd --uid 10001 --gid ugsl --no-create-home --shell /usr/sbin/nologin ugsl \
    && install -d -o ugsl -g ugsl -m 0700 /var/tmp/ugsl

WORKDIR /app
# Positive COPY allowlist excludes credentials/media even if ignore rules regress.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir . \
    && python -c "from ugsl_ai_coach.assets import model_paths; model_paths()" \
    && python -c "import cv2, mediapipe, psycopg, boto3, prometheus_client" \
    && pip check

USER 10001:10001
EXPOSE 8000
# Render's dockerCommand overrides this argv for each separate process.
CMD ["python", "-m", "ugsl_ai_coach.deployment", "api"]
