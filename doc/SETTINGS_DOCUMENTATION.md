# Settings Configuration Documentation

This document provides comprehensive information about all configuration settings available in the Pneumonia Detection API.

## Application Settings

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `app_name` | str | "Pneumonia Detection API" | `APP_NAME` | Application name displayed in API docs | "Pneumonia Detection API", "Medical AI API" |
| `app_version` | str | "3.4.2" | `APP_VERSION` | API version for documentation and headers | "3.4.2", "1.0.0" |
| `debug` | bool | False | `DEBUG` | Enable debug mode for development | `True`, `False` |

## Server Configuration

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `host` | str | "0.0.0.0" | `HOST` | Server bind address | "0.0.0.0", "127.0.0.1", "localhost" |
| `port` | int | 8000 | `PORT` | Server port number | `8000`, `3000`, `80` |

## Security Settings

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `trusted_hosts` | List[str] | ["*.railway.app", "localhost", "127.0.0.1"] | `TRUSTED_HOSTS` | Allowed host headers for security | `["*.railway.app", "mydomain.com"]` |
| `cors_origins` | List[str] | ["*"] | `CORS_ORIGINS` | CORS allowed origins | `["https://myapp.com", "http://localhost:3000"]` |

## File Upload Configuration

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `max_file_size` | int | 10485760 (10 MB) | `MAX_FILE_SIZE` | Maximum file upload size in bytes | `5242880` (5MB), `20971520` (20MB) |
| `allowed_extensions` | List[str] | [".jpg", ".jpeg", ".png"] | `ALLOWED_EXTENSIONS` | Permitted file extensions | `[".jpg", ".png", ".bmp", ".tiff"]` |

## Model Configuration

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `model_path` | str | "models/pneumonia_model_standard.onnx" | `MODEL_PATH` | Path to standard model file | "models/my_model.onnx", "/opt/models/pneumonia.onnx" |
| `model_stats_path` | str | "models/model_stats_standard.json" | `MODEL_STATS_PATH` | Path to model statistics file | "models/stats.json", "/opt/models/stats.json" |
| `model_path_efficientnet_b0` | str | "models/pneumonia_model_efficientnet_b0.onnx" | `MODEL_PATH_EFFICIENTNET_B0` | Path to EfficientNet B0 model | "models/efficientnet.onnx" |
| `model_stats_path_efficientnet_b0` | str | "models/model_stats_efficientnet_b0.json" | `MODEL_STATS_PATH_EFFICIENTNET_B0` | Path to EfficientNet B0 stats | "models/efficientnet_stats.json" |


## Advanced Rate Limiting - Core Settings

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `advanced_rate_limiting_enabled` | bool | True | `ADVANCED_RATE_LIMITING_ENABLED` | Enable/disable advanced rate limiting system | `True`, `False` |
| `max_requests_per_ip` | int | 10 | `MAX_REQUESTS_PER_IP` | Maximum requests per IP per window | `5`, `20`, `50` |
| `max_fingerprint_requests` | int | 3 | `MAX_FINGERPRINT_REQUESTS` | Maximum requests per unique fingerprint | `1`, `5`, `10` |
| `rate_limit_window_size` | int | 60 | `RATE_LIMIT_WINDOW_SIZE` | Rate limiting window size in seconds | `30`, `120`, `300` |

## Attack Detection Thresholds

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `ip_switching_threshold` | int | 3 | `IP_SWITCHING_THRESHOLD` | Same fingerprint from N+ IPs triggers detection | `2`, `5`, `10` |
| `suspicious_ip_changes_threshold` | int | 5 | `SUSPICIOUS_IP_CHANGES_THRESHOLD` | Distributed attack detection threshold | `3`, `10`, `20` |
| `global_attack_threshold` | float | 0.8 | `GLOBAL_ATTACK_THRESHOLD` | Global attack score threshold (0.0-1.0) | `0.6`, `0.7`, `0.9` |
| `bot_behavior_variance` | float | 0.1 | `BOT_BEHAVIOR_VARIANCE` | Request timing variance for bot detection | `0.05`, `0.2`, `0.3` |

## Block Durations

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `attack_block_duration` | int | 300 | `ATTACK_BLOCK_DURATION` | Block duration after attack detection (seconds) | `60`, `600`, `1800` |
| `fingerprint_block_duration` | int | 300 | `FINGERPRINT_BLOCK_DURATION` | Block duration for fingerprint violations (seconds) | `120`, `900`, `3600` |

## Detection Windows

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `ip_switching_detection_window` | int | 300 | `IP_SWITCHING_DETECTION_WINDOW` | Time window for IP switching analysis (seconds) | `180`, `600`, `900` |
| `behavioral_analysis_window` | int | 300 | `BEHAVIORAL_ANALYSIS_WINDOW` | Time window for behavioral pattern analysis (seconds) | `120`, `600`, `1200` |
| `global_attack_score_window` | int | 60 | `GLOBAL_ATTACK_SCORE_WINDOW` | Time window for global attack scoring (seconds) | `30`, `120`, `300` |

## Attack Detection Limits

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `coordinated_attack_threshold` | int | 3 | `COORDINATED_ATTACK_THRESHOLD` | Same file from N+ IPs triggers detection | `2`, `5`, `8` |
| `bot_timing_threshold` | float | 5.0 | `BOT_TIMING_THRESHOLD` | Average interval threshold for bot detection (seconds) | `1.0`, `10.0`, `30.0` |

## In-Memory Storage Limits

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `max_recent_ips` | int | 1000 | `MAX_RECENT_IPS` | Maximum recent IPs to track in memory | `500`, `2000`, `5000` |
| `max_request_patterns_per_ip` | int | 10 | `MAX_REQUEST_PATTERNS_PER_IP` | Maximum request patterns per IP to store | `5`, `20`, `50` |
| `max_file_hash_requests` | int | 100 | `MAX_FILE_HASH_REQUESTS` | Maximum file hash requests to track | `50`, `200`, `500` |
| `max_global_request_rate` | int | 1000 | `MAX_GLOBAL_REQUEST_RATE` | Maximum global request rate per window | `500`, `2000`, `10000` |
| `max_fingerprints_per_ip` | int | 100 | `MAX_FINGERPRINTS_PER_IP` | Maximum fingerprints per IP to track | `50`, `200`, `500` |

## Performance Settings

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `attack_reduction_factor` | float | 0.5 | `ATTACK_REDUCTION_FACTOR` | Factor to reduce limits during attacks (0.0-1.0) | `0.3`, `0.7`, `0.9` |

## Storage Backend Configuration

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `storage_backend` | str | "memory" | `STORAGE_BACKEND` | Storage backend type | "memory", "redis", "database" |

## Redis Configuration

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `redis_host` | str | "localhost" | `REDIS_HOST` | Redis server hostname | "localhost", "redis.example.com", "127.0.0.1" |
| `redis_port` | int | 6379 | `REDIS_PORT` | Redis server port | `6379`, `6380`, `16379` |
| `redis_password` | Optional[str] | None | `REDIS_PASSWORD` | Redis authentication password | `"mypassword"`, `None` |
| `redis_db` | int | 0 | `REDIS_DB` | Redis database number | `0`, `1`, `15` |
| `redis_url` | Optional[str] | None | `REDIS_URL` | Complete Redis connection URL | `"redis://localhost:6379/0"`, `"redis://user:pass@host:port/db"` |
| `redis_max_connections` | int | 50 | `REDIS_MAX_CONNECTIONS` | Maximum Redis connection pool size | `10`, `100`, `200` |
| `redis_cluster_mode` | bool | False | `REDIS_CLUSTER_MODE` | Enable Redis cluster mode | `True`, `False` |
| `redis_cluster_nodes` | Optional[str] | None | `REDIS_CLUSTER_NODES` | JSON string of Redis cluster nodes | `'[{"host": "node1", "port": 7000}]'` |
| `redis_key_prefix` | str | "pneumonia_api:rate_limit:" | `REDIS_KEY_PREFIX` | Prefix for all Redis keys | `"myapp:"`, `"prod:api:"` |

## Memory Storage Configuration

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `memory_max_size` | int | 10000 | `MEMORY_MAX_SIZE` | Maximum items in memory storage | `5000`, `20000`, `50000` |
| `memory_cleanup_interval` | int | 300 | `MEMORY_CLEANUP_INTERVAL` | Memory cleanup interval in seconds | `60`, `600`, `1800` |
| `memory_default_ttl` | int | 3600 | `MEMORY_DEFAULT_TTL` | Default TTL for memory items (seconds) | `1800`, `7200`, `86400` |

## Cache Configuration

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `cache_duration` | int | 300 | `CACHE_DURATION` | Cache duration in seconds | `60`, `600`, `1800` |
| `file_hash_cache_max_size` | int | 5000 | `FILE_HASH_CACHE_MAX_SIZE` | Maximum file hashes to cache | `1000`, `10000`, `20000` |

## Logging Configuration

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `log_level` | str | "INFO" | `LOG_LEVEL` | Logging level | "DEBUG", "INFO", "WARNING", "ERROR" |
| `log_format` | str | "%(asctime)s - %(name)s - %(levelname)s - %(message)s" | `LOG_FORMAT` | Log message format | Custom format strings |

## Deployment Settings

| Setting | Type | Default Value | Environment Variable | Description | Example Values |
|---------|------|---------------|----------------------|-------------|----------------|
| `railway_environment` | Optional[str] | None | `RAILWAY_ENVIRONMENT` | Railway deployment environment | "production", "staging", "development" |

## Configuration Usage Examples

### Environment Variables Setup

```bash
# Basic Configuration
export APP_NAME="My Pneumonia API"
export DEBUG=true
export PORT=3000

# Rate Limiting Configuration
export MAX_REQUESTS_PER_IP=20
export IP_SWITCHING_THRESHOLD=5
export ATTACK_BLOCK_DURATION=600

# Redis Configuration
export STORAGE_BACKEND=redis
export REDIS_HOST=my-redis-server.com
export REDIS_PASSWORD=mypassword
export REDIS_DB=1
```

### .env File Example

```env
# Application Settings
APP_NAME=Pneumonia Detection API
APP_VERSION=3.4.2
DEBUG=false

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Rate Limiting
ADVANCED_RATE_LIMITING_ENABLED=true
MAX_REQUESTS_PER_IP=15
MAX_FINGERPRINT_REQUESTS=5
IP_SWITCHING_THRESHOLD=4
ATTACK_BLOCK_DURATION=900

# Storage Backend
STORAGE_BACKEND=memory
MEMORY_MAX_SIZE=15000

# Security
TRUSTED_HOSTS=["*.mydomain.com", "localhost"]
CORS_ORIGINS=["https://myapp.com", "http://localhost:3000"]
```

### Programmatic Access

```python
from app.core.settings import settings

# Get individual settings
max_requests = settings.max_requests_per_ip
block_duration = settings.attack_block_duration

# Get rate limiting configuration dict
rate_config = settings.get_rate_limiting_config()
print(f"Max requests per IP: {rate_config['max_requests_per_ip']}")

# Get storage configuration
storage_config = settings.get_storage_config()

# Get Redis configuration
redis_config = settings.get_redis_config()
```

## Configuration Methods

The Settings class provides several helper methods:

### `get_rate_limiting_config() -> dict`
Returns a comprehensive dictionary containing all rate limiting configuration parameters.

### `get_storage_config() -> dict`
Returns storage backend configuration based on the selected `storage_backend`.

### `get_redis_config() -> dict`
Returns Redis-specific configuration including cluster support.
