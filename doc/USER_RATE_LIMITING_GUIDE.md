# JWT User-Based Rate Limiting

## Overview

This document describes the simplified JWT user-based rate limiting system that replaces the advanced multi-layer rate limiting when JWT authentication is enabled.

## Architecture

```
JWT Identity → Rate Limit per User → Store Counters in Supabase
```

### Key Changes from Advanced Rate Limiting

| Feature | Advanced (IP-based) | User-based (JWT) |
|---------|-------------------|------------------|
| Identity | Client IP + Fingerprint | JWT user_id (sub claim) |
| Rate Limit | Per IP & fingerprint | Per user account |
| Storage | In-memory | Supabase (with memory fallback) |
| Attack Detection | Yes (complex) | No (JWT auth provides identity) |
| Configuration | 30+ settings | 4 simple settings |

## Configuration

### Environment Variables

```env
# JWT authentication is always enabled (native)

# Supabase configuration (required)
SUPABASE_URL=https://your-project.supabase.co

# User rate limiting settings
USER_RATE_LIMITING_ENABLED=true
USER_RATE_LIMIT_MAX_REQUESTS=100
USER_RATE_LIMIT_WINDOW_SECONDS=3600
```

### Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `user_rate_limiting_enabled` | `true` | Enable per-user rate limiting |
| `user_rate_limit_max_requests` | `100` | Max requests per user per window |
| `user_rate_limit_window_size` | `3600` | Window size in seconds (1 hour) |
| `user_rate_limit_use_supabase` | `true` | Use Supabase for storage |

## Supabase Setup

### 1. Create the Rate Limits Table

Run the SQL script in Supabase SQL Editor:

```sql
CREATE TABLE rate_limits (
    user_id TEXT PRIMARY KEY,
    request_count INTEGER DEFAULT 0,
    window_start DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE rate_limits ENABLE ROW LEVEL SECURITY;

-- Allow API access
CREATE POLICY "Allow anon insert" ON rate_limits FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "Allow anon select" ON rate_limits FOR SELECT TO anon USING (true);
CREATE POLICY "Allow anon update" ON rate_limits FOR UPDATE TO anon USING (true) WITH CHECK (true);
```

See `doc/SUPABASE_RATE_LIMITS_TABLE.sql` for the complete SQL script.

### 2. Configure API Keys

- Use the **anon key** for public API access
- Use the **service role key** for administrative operations

## How It Works

### Request Flow

1. **Request received** → Middleware extracts `Authorization: Bearer <token>`
2. **JWT verified** → Extracts `user_id` from `sub` claim
3. **Rate limit check** → Query/increment counter in Supabase
4. **Decision** → Allow or reject based on counter vs limit

### Rate Limit Algorithm

- **Sliding Window**: Each user has a window starting from their first request
- **Counter Reset**: When `window_start + window_size < current_time`, counter resets
- **Atomic Increment**: Counter incremented atomically per request

### Fallback Behavior

1. **No JWT token** → Falls back to IP-based rate limiting
2. **Invalid JWT** → Falls back to IP-based rate limiting  
3. **Supabase unavailable** → Falls back to in-memory storage
4. **Rate limit check fails** → Request allowed (fail-open)

## Response Headers

All responses include rate limiting headers:

```http
X-RateLimit-Limit: 100 per 3600s
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699999999
X-RateLimit-Window: 3600
X-RateLimit-Type: user
```

### Rate Limited Response (429)

```json
{
    "error": "Rate limit exceeded",
    "message": "User rate limit exceeded (100/100)",
    "client_ip": "1.2.3.4",
    "endpoint": "POST /pneumonia/predict",
    "timestamp": 1699999999.123,
    "details": {
        "user_id": "a1b2c3d4...",
        "requests_made": 100,
        "requests_limit": 100,
        "window_size": 3600,
        "retry_after": 1800
    }
}
```

## Code Examples

### Python Client

```python
import httpx

async def make_request(token: str, image_path: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://your-api.com/pneumonia/predict",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": open(image_path, "rb")}
        )
        
        # Check rate limit headers
        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        if remaining < 10:
            print(f"Warning: Only {remaining} requests remaining")
        
        return response.json()
```

### JavaScript/TypeScript

```typescript
async function predictPneumonia(token: string, file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/pneumonia/predict', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
    });
    
    if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After');
        throw new Error(`Rate limited. Retry after ${retryAfter} seconds`);
    }
    
    return response.json();
}
```

## Monitoring & Debugging

### Check User Rate Limit Status

```python
from app.core.user_rate_limiting import get_user_rate_limiter

limiter = get_user_rate_limiter()
if limiter:
    status = await limiter.get_user_status(user_id)
    print(f"Requests: {status.requests_made}/{status.requests_limit}")
    print(f"Remaining: {status.requests_remaining}")
    print(f"Reset in: {status.retry_after} seconds")
```

### Enable Debug Headers

Set `RATE_LIMIT_DEBUG_HEADERS=true` for additional debugging information.

## Migration from Advanced Rate Limiting

The advanced rate limiting (IP + fingerprint + attack detection) is preserved as a backup:

1. **Backup location**: `backup_advanced_rate_limiting/`
2. **Legacy support**: IP-based rate limiting still works when JWT auth is disabled
3. **Gradual migration**: Both systems can coexist during transition

### Recommended Migration Steps

1. Enable JWT authentication in staging
2. Test user rate limiting with low limits (10 requests/minute)
3. Monitor Supabase table for correctness
4. Gradually increase limits to production values
5. Enable in production

## Troubleshooting

### Rate Limiter Not Initialized

```
WARNING: User rate limiter not initialized
```

**Causes**:
- `USER_RATE_LIMITING_ENABLED=false`
- Supabase connection failed

**Solution**: Check `get_status()` for initialization state.

### Supabase Connection Failed

```
WARNING: Failed to initialize Supabase storage, using in-memory
```

**Causes**:
- Invalid `SUPABASE_URL`
- Invalid `SUPABASE_ANON_KEY`
- Network issues
- RLS policies blocking access

**Solution**: Test connection manually with curl.

### Rate Limit Table Not Found

```
ERROR: Supabase GET failed: 404
```

**Cause**: `rate_limits` table doesn't exist.

**Solution**: Run the SQL script in Supabase SQL Editor.
