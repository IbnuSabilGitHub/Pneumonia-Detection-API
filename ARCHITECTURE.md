# Pneumonia Detection API - Architecture v3.8.0

## Overview

Version 3.8.0 simplifies the API around one production path:

```text
Supabase JWT authentication -> per-user rate limiting -> pneumonia prediction
```

The previous IP/fingerprint-based advanced rate limiting system has been removed. The API now trusts the Supabase JWT as the source of user identity, extracts the `sub` claim, and applies request limits per authenticated user. Admin endpoints (`/stats` and `/status`) were also removed.

## Architecture Principles

- Keep authentication and quota decisions tied to a stable user identity.
- Prefer a small, auditable security surface over complex behavioral detection.
- Use FastAPI dependency injection for testable services.
- Keep model inference isolated in the service layer.
- Fail open for rate-limit storage outages so legitimate users are not blocked by Supabase REST failures.
- Keep public monitoring simple through `/health` and `/badge.json`.

## Major v3.8.0 Changes

- Removed `app/core/rate_limiting/` and the old storage factory implementation.
- Removed IP switching detection, request fingerprint blocking, behavioral analysis, and global attack scoring.
- Removed admin route modules from the running application.
- Added `app/core/user_rate_limiting.py`.
- Added Supabase-backed rate-limit counters with in-memory fallback.
- Kept `app/core/advanced_rate_limiting.py` as a compatibility shim that returns `None`.
- Updated the active API surface to health, prediction, model info, and generated docs.

## Runtime Components

```text
app.main:create_app()
|-- creates FastAPI app with lifespan manager
|-- registers middleware through MiddlewareFactory
|-- includes health router
|-- includes prediction router at /pneumonia
|-- includes model_info router at /pneumonia
`-- installs custom exception handlers
```

Startup initializes:

- `PneumoniaPredictionService`
- `UserRateLimiter` when user rate limiting is enabled
- dependency container values for route handlers
- background cleanup for duplicate-file hash cache

Shutdown cleans up:

- user rate limiter resources, including the Supabase HTTP client
- prediction service resources when available
- in-memory service references

## Project Structure

```text
pneumonia-detection-api/
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |-- openapi.py
|   |-- version.py
|   |-- api/
|   |   |-- __init__.py
|   |   |-- health.py
|   |   |-- model_info.py
|   |   `-- prediction.py
|   |-- core/
|   |   |-- __init__.py
|   |   |-- advanced_rate_limiting.py
|   |   |-- dependencies.py
|   |   |-- logger.py
|   |   |-- middleware_factory.py
|   |   |-- settings.py
|   |   |-- startup_manager.py
|   |   `-- user_rate_limiting.py
|   |-- docs/
|   |   |-- base_builder.py
|   |   `-- sections/
|   |       |-- api_metadata.py
|   |       |-- health_metadata.py
|   |       |-- model_info_metadata.py
|   |       `-- prediction_metadata.py
|   |-- middleware/
|   |   |-- __init__.py
|   |   `-- security.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- auth_schemas.py
|   |   |-- base.py
|   |   |-- error_codes.py
|   |   |-- health_schemas.py
|   |   |-- model_info_schemas.py
|   |   |-- prediction_schemas.py
|   |   `-- security_schemes.py
|   |-- services/
|   |   |-- __init__.py
|   |   `-- prediction.py
|   `-- utils/
|       |-- __init__.py
|       |-- auth.py
|       |-- exceptions.py
|       |-- get_prediction_service.py
|       |-- jwt_auth.py
|       |-- security.py
|       `-- validation.py
|-- doc/
|   |-- SUPABASE_JWT_AUTH.md
|   |-- USER_RATE_LIMITING_GUIDE.md
|   `-- SUPABASE_RATE_LIMITS_TABLE.sql
|-- models/
|-- main.py
|-- Dockerfile
|-- docker-compose.yml
|-- render.yaml
|-- requirements.txt
`-- README.md
```

## Layer Architecture

### API Layer (`app/api/`)

The API layer owns HTTP request handling and response construction.

- `health.py`: exposes `/`, `/health`, `HEAD` probes, and `/badge.json`.
- `prediction.py`: exposes `POST /pneumonia/predict`, requires Supabase JWT, validates uploads, applies concurrency control, and delegates inference to the service layer.
- `model_info.py`: exposes `GET /pneumonia/model/info` for loaded model metadata.

Removed from the active app:

- `/stats`
- `/status`
- `/security/stats`
- `/security/status`

### Middleware Layer (`app/middleware/`)

`SecurityMiddleware` handles cross-cutting request behavior:

- request logging
- security headers
- excluded-path checks
- JWT user extraction for rate limiting
- per-user rate-limit checks
- `429` response construction

When a valid JWT is present and user rate limiting is enabled, middleware checks `UserRateLimiter`. If JWT extraction or the user limiter is unavailable, the old advanced limiter path is consulted, but the compatibility shim currently returns `None`, so the request is allowed to continue.

### Core Layer (`app/core/`)

The core layer owns application infrastructure:

- `settings.py`: Pydantic settings loaded from environment variables and `.env`.
- `startup_manager.py`: initializes prediction and user rate-limit services.
- `user_rate_limiting.py`: implements per-user limits with Supabase storage and memory fallback.
- `dependencies.py`: stores initialized service instances for FastAPI dependencies.
- `middleware_factory.py`: composes middleware in one place.
- `advanced_rate_limiting.py`: compatibility shim for imports that still reference the old system.

### Service Layer (`app/services/`)

`PneumoniaPredictionService` owns ML behavior:

- load ONNX model files
- preprocess uploaded images
- run inference
- calculate probabilities and confidence
- return normalized prediction output
- expose model metadata

### Utils Layer (`app/utils/`)

Utility modules provide reusable support:

- `jwt_auth.py`: Supabase ES256 JWT verification through JWKS.
- `validation.py`: file, image integrity, and image content validation.
- `security.py`: client IP helpers, SHA-256 file hashing, duplicate-file cache.
- `get_prediction_service.py`: route dependency helper.
- `exceptions.py`: domain exceptions for prediction and validation failures.

### Models Layer (`app/models/`)

Pydantic models define API contracts for:

- prediction responses
- health responses
- model information
- auth-related payloads
- standardized error codes

## Request Flow

```text
Client
  |
  | Authorization: Bearer <Supabase access token>
  v
SecurityMiddleware
  |
  | verify JWT enough to identify user for rate limiting
  | increment user counter
  v
FastAPI dependency: get_current_user
  |
  | verify JWT for protected route access
  v
Prediction endpoint
  |
  | validate file metadata, size, duplicate hash, and image content
  | acquire prediction semaphore
  v
PneumoniaPredictionService
  |
  | preprocess image and run ONNX inference
  v
PredictionResponse
```

## Authentication Architecture

`POST /pneumonia/predict` requires:

```http
Authorization: Bearer <supabase_access_token>
```

JWT verification:

- fetches public keys from `SUPABASE_URL/auth/v1/.well-known/jwks.json`
- accepts ES256 Supabase tokens
- verifies expiry
- optionally verifies the `aud` claim
- requires `sub` and `exp`
- exposes `JWTPayload.user_id` as the Supabase `sub` claim

Admin helpers still exist in `app/utils/jwt_auth.py` for compatibility, but admin endpoints are no longer registered by the application.

## User Rate Limiting Architecture

The current limiter is deliberately small:

```text
JWT sub claim -> user counter -> sliding window -> allow or reject
```

Implementation:

- `UserRateLimiter` coordinates checks.
- `RateLimitInfo` carries request count, remaining quota, reset time, and retry time.
- `SupabaseRateLimitStorage` persists counters through Supabase REST.
- `InMemoryRateLimitStorage` is the fallback when Supabase is disabled or unavailable.

Default behavior:

```text
100 requests per authenticated user per 3600 seconds
```

Configuration:

| Setting | Default | Description |
|---|---:|---|
| `USER_RATE_LIMITING_ENABLED` | `true` | Enable JWT user-based rate limiting |
| `USER_RATE_LIMIT_MAX_REQUESTS` | `100` | Max requests per user per window |
| `USER_RATE_LIMIT_WINDOW_SIZE` | `3600` | Window duration in seconds |
| `USER_RATE_LIMIT_USE_SUPABASE` | `true` | Store counters in Supabase when configured |
| `SUPABASE_URL` | unset | Supabase project URL |
| `SUPABASE_ANON_KEY` | unset | Supabase key used for REST access |

Rate-limit response headers:

```http
X-RateLimit-Limit: 100 per 3600s
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1713340000
X-RateLimit-Window: 3600
X-RateLimit-Type: user
Retry-After: 1800
```

## Supabase Rate-Limit Storage

Supabase storage expects a `rate_limits` table. The schema is maintained in:

```text
doc/SUPABASE_RATE_LIMITS_TABLE.sql
```

Conceptual columns:

| Column | Purpose |
|---|---|
| `user_id` | Supabase user ID from JWT `sub` |
| `request_count` | Number of requests in the current window |
| `window_start` | Unix timestamp for the active window |
| `updated_at` | Last update timestamp |

If Supabase REST requests fail, the limiter falls back to in-memory counters. If both Supabase and memory checks fail, the limiter allows the request and logs the error.

## Prediction Protection

The prediction endpoint has several protections after authentication:

- file extension validation for `.jpg`, `.jpeg`, and `.png`
- max file size validation, default `10 MB`
- image integrity validation
- chest X-ray content validation
- duplicate upload detection through SHA-256 hash cache
- semaphore-based concurrency control, default `3`
- structured error responses with stable error codes

The concurrency semaphore protects CPU-bound ONNX inference from request bursts. It is separate from user rate limiting.

## Public Monitoring

Supported monitoring endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Detailed health status with model-loaded state, version, and uptime |
| `HEAD /health` | Lightweight platform health probe |
| `GET /badge.json` | Shields.io-compatible badge payload |
| `GET /` | Root health check |

The removed `/stats` and `/status` endpoints should not be used by monitoring clients.

## Error Handling

The application uses:

- route-level validation errors for prediction-specific failures
- custom exception handlers for `413`, `429`, and `404`
- CORS-friendly JSON error responses
- stable error code fields such as `FILE_TOO_LARGE`, `INVALID_FILE_FORMAT`, `DUPLICATE_FILE`, `MODEL_NOT_LOADED`, and `SERVICE_UNAVAILABLE`

Examples:

```json
{
  "detail": "File size exceeds limit of 10.0 MB",
  "error_code": "FILE_TOO_LARGE",
  "timestamp": "2026-04-17T00:00:00.000Z"
}
```

```json
{
  "error": "Rate limit exceeded",
  "message": "User rate limit exceeded (100/100)",
  "endpoint": "POST /pneumonia/predict",
  "details": {
    "requests_made": 100,
    "requests_limit": 100,
    "window_size": 3600,
    "retry_after": 1200
  }
}
```

## Deployment Architecture

The application runs as a single FastAPI service:

```text
Web process -> FastAPI -> ONNX Runtime -> local model files
```

Primary deployment target:

- Render via `render.yaml`

Also supported:

- Docker
- Docker Compose
- local Python virtual environment

External services:

- Supabase Auth for JWT issuance and JWKS
- Supabase REST for optional persistent rate-limit counters

Redis is no longer part of the architecture.

## Configuration Reference

| Variable | Default | Description |
|---|---:|---|
| `APP_NAME` | `Pneumonia Detection API` | Application name |
| `APP_VERSION` | code/env value | Application version |
| `DEBUG` | `false` | Enable reload/debug behavior |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |
| `SUPABASE_URL` | unset | Required for JWT verification |
| `SUPABASE_ANON_KEY` | unset | Used for Supabase rate-limit storage |
| `SUPABASE_JWT_VERIFY_AUDIENCE` | `true` | Verify JWT `aud` claim |
| `USER_RATE_LIMITING_ENABLED` | `true` | Enable per-user limits |
| `USER_RATE_LIMIT_MAX_REQUESTS` | `100` | Requests per user per window |
| `USER_RATE_LIMIT_WINDOW_SIZE` | `3600` | User rate-limit window in seconds |
| `USER_RATE_LIMIT_USE_SUPABASE` | `true` | Prefer Supabase storage |
| `PREDICTION_CONCURRENCY_LIMIT` | `3` | Max concurrent predictions |
| `MAX_FILE_SIZE` | `10485760` | Max upload size in bytes |
| `ALLOWED_EXTENSIONS` | `.jpg,.jpeg,.png` | Allowed upload extensions |
| `CACHE_DURATION` | `300` | Duplicate-file hash cache TTL |
| `FILE_HASH_CACHE_MAX_SIZE` | `200` | Duplicate-file hash cache capacity |
| `MODEL_PATH` | `models/pneumonia_model_standard.onnx` | Standard model path |
| `MODEL_STATS_PATH` | `models/model_stats_standard.json` | Standard model stats |
| `MODEL_PATH_EFFICIENTNET_B0` | `models/pneumonia_model_efficientnet_b0.onnx` | EfficientNet-B0 model path |
| `MODEL_STATS_PATH_EFFICIENTNET_B0` | `models/model_stats_efficientnet_b0.json` | EfficientNet-B0 stats |
| `TRUSTED_HOSTS` | `*.onrender.com,localhost,127.0.0.1` | Trusted host list |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `EXCLUDED_PATHS` | `/health,/,/docs,/redoc,/openapi.json` | Paths excluded from rate limiting |
| `LOG_LEVEL` | `INFO` | Logging level |

## Migration Notes

For clients upgrading to v3.8.0:

- Replace `/stats` and `/status` monitoring with `/health`.
- Stop sending `X-Admin-API-Key`; admin endpoints are not active.
- Always send Supabase JWT Bearer tokens to `POST /pneumonia/predict`.
- Expect rate limits to follow the user account, not IP address.
- Prepare the Supabase `rate_limits` table if persistent quota storage is desired.
- Remove Redis configuration from deployment environments.

## Future Direction

The architecture now leaves room for simpler, user-aware product features:

- account-level plans and quotas
- per-user usage dashboards
- organization-level quotas
- audit logs keyed by Supabase user ID
- optional persistent analytics built on Supabase tables

These should build on JWT identity rather than reintroducing IP/fingerprint heuristics.
