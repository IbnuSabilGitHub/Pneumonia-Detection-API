# Multi-stage build for Pneumonia Detection API v3.4.2
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Create venv + install deps
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_VERSION=3.4.2 \
    STORAGE_BACKEND=memory

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