# Multi-stage build for Pneumonia Detection API v3.6.0
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Create venv + install deps
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
        APP_VERSION=3.6.0 \
        DEBUG=false \
        ALLOWED_ORIGINS=http://localhost:3000 \
        STORAGE_BACKEND="memory" \
        ADVANCED_RATE_LIMITING_ENABLED=true \
        MAX_REQUESTS_PER_IP=100 \
        MAX_FINGERPRINT_REQUESTS=50 \
        RATE_LIMIT_WINDOW_SIZE=300 \
        IP_SWITCHING_THRESHOLD=5 \
        SUSPICIOUS_IP_CHANGES_THRESHOLD=10 \
        GLOBAL_ATTACK_THRESHOLD=0.7 \
        BOT_BEHAVIOR_VARIANCE=0.1 \
        ATTACK_BLOCK_DURATION=300 \
        FINGERPRINT_BLOCK_DURATION=600 \
        IP_SWITCHING_DETECTION_WINDOW=300 \
        BEHAVIORAL_ANALYSIS_WINDOW=600 \
        GLOBAL_ATTACK_SCORE_WINDOW=900 \
        COORDINATED_ATTACK_THRESHOLD=3 \
        BOT_TIMING_THRESHOLD=2.0 \
        MAX_RECENT_IPS=1000 \
        MAX_REQUEST_PATTERNS_PER_IP=50 \
        MAX_FILE_HASH_REQUESTS=500 \
        MAX_GLOBAL_REQUEST_RATE=500 \
        MAX_FINGERPRINTS_PER_IP=10 \
        ATTACK_REDUCTION_FACTOR=0.5 \
        PREDICTION_CONCURRENCY_LIMIT=3 \
        PREDICTION_MAX_REQUESTS_PER_IP=18 \
        PREDICTION_RATE_WINDOW=300 \
        EXCLUDED_PATHS="/health,/,/docs,/redoc,/openapi.json" \
        LOG_LEVEL=INFO \
        LOG_ENABLED=true \
        LOG_INCLUDE_TIMESTAMP=false \
        LOG_INCLUDE_LEVEL=true \
        LOG_INCLUDE_LOGGER_NAME=true \
        LOG_FIELD_SEPARATOR=" - " \
        LOG_FORMAT_WITH_TIMESTAMP="%(asctime)s - %(levelname)s - %(message)s" \
        LOG_FORMAT_WITHOUT_TIMESTAMP="%(levelname)s - %(message)s" \
        MEMORY_MAX_SIZE=1000 \
        MEMORY_CLEANUP_INTERVAL=180 \
        MEMORY_DEFAULT_TTL=1800 \
        TRUSTED_HOSTS="localhost,127.0.0.1,0.0.0.0" \
        CORS_ORIGINS="*" \
        ALLOWED_EXTENSIONS=".jpg,.jpeg,.png"


WORKDIR /app

# Non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Copy venv
COPY --from=builder /opt/venv /opt/venv

# Copy app
COPY --chown=app:app main.py ./
COPY --chown=app:app app/ ./app/
COPY --chown=app:app models/ ./models/

USER app
EXPOSE 8000

# Uvicorn langsung (lebih ringan daripada python main.py jika main hanya wrapper)
CMD ["python", "main.py"]