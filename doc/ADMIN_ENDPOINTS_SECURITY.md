# Admin Endpoints Security Guide

## Overview

Endpoint `/stats` dan `/status` sekarang dilindungi dengan authentication untuk mencegah information disclosure dan attack intelligence gathering.

## Security Configuration

### Environment Variables

```bash
# REQUIRED: Set admin API key untuk akses /stats dan /status
ADMIN_API_KEY=your-secure-random-key-here

# OPTIONAL: Enable public access (NOT RECOMMENDED for production)
ENABLE_PUBLIC_STATS=false  # Set true untuk allow public access ke /stats
ENABLE_PUBLIC_STATUS=false # Set true untuk allow public access ke /status
```

### Generate Secure API Key

```bash
# Linux/Mac
openssl rand -hex 32

# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# PowerShell
[Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

## Usage Examples

### ✅ Authenticated Request (Recommended)

```bash
# Request with API key
curl -H "X-Admin-API-Key: your-secret-key" https://your-api.com/stats

# Response: Success
{
  "service": "Pneumonia Detection API",
  "security_metrics": {
    "global_attack_score": 0.15,
    "requests_per_minute": 45,
    ...
  }
}
```

### ❌ Unauthenticated Request (Blocked)

```bash
# Request without API key
curl https://your-api.com/stats

# Response: 401 Unauthorized
{
  "error": "Missing API key",
  "message": "Admin endpoints require X-Admin-API-Key header",
  "required_header": "X-Admin-API-Key"
}
```

### ❌ Invalid API Key (Blocked)

```bash
# Request with wrong API key
curl -H "X-Admin-API-Key: wrong-key" https://your-api.com/stats

# Response: 403 Forbidden
{
  "error": "Invalid API key",
  "message": "The provided API key is not valid"
}
```

## Endpoint Access Matrix

| Endpoint | Public Access | Admin Access | Purpose |
|----------|---------------|--------------|---------|
| `/health` | ✅ Yes | ✅ Yes | Service health check |
| `/predict` | ✅ Yes (rate limited) | ✅ Yes | Pneumonia detection |
| `/model/info` | ✅ Yes | ✅ Yes | Model metadata |
| `/stats` | ❌ No (default) | ✅ Yes (with API key) | Security analytics |
| `/status` | ❌ No (default) | ✅ Yes (with API key) | Security status |

## Security Best Practices

### 1. API Key Management

```bash
# ✅ GOOD: Set in environment variable
export ADMIN_API_KEY="$(openssl rand -hex 32)"

# ❌ BAD: Hardcode in code
ADMIN_API_KEY = "my-secret-key"  # Don't do this!

# ❌ BAD: Use simple password
ADMIN_API_KEY = "admin123"  # Too weak!
```

### 2. Production Deployment

```yaml
# docker-compose.yml
services:
  pneumonia-api:
    environment:
      # Generate unique key per environment
      ADMIN_API_KEY: ${ADMIN_API_KEY}  # From .env file
      ENABLE_PUBLIC_STATS: "false"
      ENABLE_PUBLIC_STATUS: "false"
```

```bash
# .env file (never commit to git!)
ADMIN_API_KEY=your-production-key-here
```

### 3. Access Control

```plaintext
WHO SHOULD HAVE ACCESS:
✅ DevOps team - for monitoring
✅ Security team - for threat analysis
✅ Incident responders - for debugging

WHO SHOULD NOT:
❌ End users - no need for internal metrics
❌ Public internet - security risk
❌ Untrusted clients - potential attack vector
```

## Migration Guide

### From Public to Protected (Breaking Change)

**Before (v3.4.3 and earlier):**
```bash
# Anyone could access
curl https://api.example.com/stats
# ✅ 200 OK - Full metrics exposed
```

**After (v3.4.4+):**
```bash
# Requires authentication
curl https://api.example.com/stats
# ❌ 401 Unauthorized

# Must use API key
curl -H "X-Admin-API-Key: YOUR_KEY" https://api.example.com/stats
# ✅ 200 OK - Secured access
```

### Temporary Public Access (Development Only)

If you need temporary public access for development:

```bash
# .env.development
ENABLE_PUBLIC_STATS=true
ENABLE_PUBLIC_STATUS=true
ADMIN_API_KEY=  # Leave empty if public access enabled
```

**⚠️ WARNING:** Never enable public access in production!

## Monitoring and Alerting

### Failed Authentication Attempts

The system logs all failed authentication attempts:

```log
2025-11-08 10:15:23 - app.utils.auth - WARNING - Invalid admin API key attempt from request
```

### Recommended Alerts

```yaml
# Example: Prometheus alert
- alert: SuspiciousAdminAccess
  expr: rate(admin_auth_failures[5m]) > 10
  annotations:
    summary: "High rate of failed admin auth attempts"
    description: "Possible brute force attack on admin endpoints"
```

## FAQ

### Q: Do I need to set ADMIN_API_KEY?

**A:** Yes, for production. If not set, `/stats` and `/status` will return 503 (Service Unavailable).

### Q: Can I use different keys for different teams?

**A:** Current implementation supports one key. For multi-key support, consider implementing API key database with roles.

### Q: What if I forget my API key?

**A:** Regenerate a new key using the methods above and update your environment variables. Old key will be invalidated immediately.

### Q: Can I use this in CI/CD?

**A:** Yes! Set `ADMIN_API_KEY` in your CI/CD secrets and use it for health checks.

```yaml
# GitHub Actions example
- name: Check API Stats
  run: |
    curl -f -H "X-Admin-API-Key: ${{ secrets.ADMIN_API_KEY }}" \
      https://api.example.com/stats
```

## Troubleshooting

### Issue: 503 Service Unavailable

**Cause:** `ADMIN_API_KEY` not configured

**Solution:**
```bash
export ADMIN_API_KEY="$(openssl rand -hex 32)"
# Restart API service
```

### Issue: 401 Unauthorized

**Cause:** Missing `X-Admin-API-Key` header

**Solution:**
```bash
# Add header to your request
curl -H "X-Admin-API-Key: your-key" https://api.example.com/stats
```

### Issue: 403 Forbidden

**Cause:** Invalid API key

**Solution:**
- Verify your API key matches the one in environment variable
- Check for trailing spaces or newlines
- Regenerate key if compromised

## Related Documentation

- [SECURITY-FEATURES.md](./SECURITY-FEATURES.md) - Complete security overview
- [ADVANCED_RATE_LIMITING_DOCS.md](./ADVANCED_RATE_LIMITING_DOCS.md) - Rate limiting details
- [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Production deployment

## Support

For security issues, contact: security@yourcompany.com
