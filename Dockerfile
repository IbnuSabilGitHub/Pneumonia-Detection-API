# Multi-stage build for Pneumonia Detection API v3.1.0
FROM python:3.11-slim as builder

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install system dependencies and security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Copy application files with proper ownership
COPY --chown=app:app main.py .
COPY --chown=app:app app/ ./app/
COPY --chown=app:app models/ ./models/

# Set environment variables for production
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_VERSION=3.1.0

# Switch to non-root user
USER app

EXPOSE 8000

# Use the updated main.py as entry point
CMD ["python", "main.py"]