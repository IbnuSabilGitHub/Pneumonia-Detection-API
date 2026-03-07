# Migration Guide: v3.5.0 → v3.5.1

## 🔒 Major Change: Admin Endpoints Authentication

**Release Date**: November 9, 2025  
**Version**: 3.5.1  
**Breaking Change**: YES

---

## 📋 Summary

Endpoints `/stats` and `/status` have been changed from **PUBLIC** to **PRIVATE** (admin-only access). This change enhances security by preventing information disclosure and attack intelligence gathering.

---

## 🎯 What Changed?

### Before (v3.5.0 and earlier)
```bash
# Anyone could access these endpoints
curl http://localhost:8000/stats
curl http://localhost:8000/status
# ✅ 200 OK - Full metrics exposed
```

### After (v3.5.1+)
```bash
# Authentication required
curl http://localhost:8000/stats
# ❌ 401 Unauthorized

curl -H "X-Admin-API-Key: YOUR_KEY" http://localhost:8000/stats
# ✅ 200 OK - Authenticated access
```

---

## 🚨 Breaking Changes

### Affected Endpoints
1. `GET /stats` - Now requires `X-Admin-API-Key` header
2. `GET /status` - Now requires `X-Admin-API-Key` header

### Who Is Affected?
- ✅ **Monitoring Systems**: Update to include API key
- ✅ **CI/CD Pipelines**: Add API key to health checks
- ✅ **Admin Dashboards**: Include authentication headers
- ✅ **Third-party Integrations**: Update API client configurations
- ❌ **End Users**: No impact (they shouldn't access these endpoints)

---

## ✅ Migration Steps

### Step 1: Generate Admin API Key

**Linux/Mac:**
```bash
openssl rand -hex 32
```

**Windows PowerShell:**
```powershell
[Convert]::ToBase64String([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
```

**Python:**
```python
import secrets
print(secrets.token_urlsafe(32))
```

**Example Output:**
```
a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456
```

### Step 2: Configure Environment

**Option A: Environment Variable (Recommended)**
```bash
export ADMIN_API_KEY="your-generated-key-here"
```

**Option B: .env File**
```bash
# .env
ADMIN_API_KEY=your-generated-key-here
ENABLE_PUBLIC_STATS=false
ENABLE_PUBLIC_STATUS=false
```

**Option C: Docker Compose**
```yaml
services:
  pneumonia-api:
    environment:
      - ADMIN_API_KEY=${ADMIN_API_KEY}
      - ENABLE_PUBLIC_STATS=false
      - ENABLE_PUBLIC_STATUS=false
```

### Step 3: Restart Service

```bash
# Docker Compose
docker-compose restart pneumonia-api

# Local Development
# Stop and restart the application

# Railway/Render
# Redeploy with new environment variables
```

### Step 4: Update Client Code

**cURL:**
```bash
# Before
curl http://localhost:8000/stats

# After
curl -H "X-Admin-API-Key: your-key" http://localhost:8000/stats
```

**Python (requests):**
```python
import requests

# Before
response = requests.get("http://localhost:8000/stats")

# After
headers = {"X-Admin-API-Key": "your-key"}
response = requests.get("http://localhost:8000/stats", headers=headers)
```

**JavaScript (fetch):**
```javascript
// Before
fetch('http://localhost:8000/stats')

// After
fetch('http://localhost:8000/stats', {
  headers: {
    'X-Admin-API-Key': 'your-key'
  }
})
```

**Node.js (axios):**
```javascript
// Before
axios.get('http://localhost:8000/stats')

// After
axios.get('http://localhost:8000/stats', {
  headers: {
    'X-Admin-API-Key': 'your-key'
  }
})
```

### Step 5: Update Monitoring Scripts

**Health Check Script:**
```bash
#!/bin/bash
# health_check.sh

ADMIN_KEY="${ADMIN_API_KEY}"
API_URL="http://localhost:8000"

# Check status
STATUS=$(curl -s -H "X-Admin-API-Key: $ADMIN_KEY" "$API_URL/status" | jq -r .security_status)

if [ "$STATUS" == "active" ]; then
  echo "✅ API is healthy"
  exit 0
else
  echo "❌ API is unhealthy: $STATUS"
  exit 1
fi
```

**CI/CD Integration (GitHub Actions):**
```yaml
# .github/workflows/deploy.yml
- name: Health Check
  env:
    ADMIN_API_KEY: ${{ secrets.ADMIN_API_KEY }}
  run: |
    curl -f -H "X-Admin-API-Key: $ADMIN_API_KEY" \
      https://your-api.com/status
```

---

## 🔄 Rollback Plan

If you need to temporarily revert to public access (NOT RECOMMENDED for production):

```bash
# .env
ENABLE_PUBLIC_STATS=true
ENABLE_PUBLIC_STATUS=true

# Restart service
docker-compose restart pneumonia-api
```

**⚠️ WARNING**: This defeats the security enhancement. Use only for development/testing.

---

## 🛡️ Security Benefits

### Why This Change?

1. **Information Disclosure Prevention**
   - Attackers cannot view real-time threat metrics
   - System performance data is hidden
   - Storage backend information protected

2. **Attack Intelligence Mitigation**
   - Prevents adversaries from monitoring detection effectiveness
   - No real-time feedback for attackers to adjust strategies
   - Blocks reconnaissance attempts

3. **System Profiling Protection**
   - Internal architecture details hidden
   - Response time metrics not exposed
   - Resource utilization data protected

4. **Industry Best Practice**
   - Aligns with GitHub, AWS, Stripe security models
   - Follows "defense in depth" principle
   - Implements "least privilege" access control

---

## 📊 Impact Analysis

### Low Impact Scenarios ✅
- **Admin Teams**: Simply add API key to existing tools
- **Internal Monitoring**: One-time configuration update
- **Development**: Use `.env` file for local testing

### Medium Impact Scenarios ⚠️
- **CI/CD Pipelines**: Update with API key from secrets
- **Third-party Monitoring**: Reconfigure with new headers
- **Dashboard Tools**: Update API client libraries

### High Impact Scenarios 🚨
- **Public Monitoring Widgets**: Remove or replace with public health endpoint
- **Unauthenticated Clients**: Require code changes
- **Legacy Integrations**: May need refactoring

---

## 🧪 Testing

### Test Authentication

```bash
# 1. Test without key (should fail)
curl -v http://localhost:8000/stats
# Expected: 401 Unauthorized

# 2. Test with wrong key (should fail)
curl -v -H "X-Admin-API-Key: wrong-key" http://localhost:8000/stats
# Expected: 403 Forbidden

# 3. Test with correct key (should succeed)
curl -v -H "X-Admin-API-Key: your-key" http://localhost:8000/stats
# Expected: 200 OK with metrics

# 4. Test public endpoint (should still work)
curl -v http://localhost:8000/health
# Expected: 200 OK without auth
```

### Verify Configuration

```bash
# Check if ADMIN_API_KEY is set
echo $ADMIN_API_KEY

# Test API key validity
curl -H "X-Admin-API-Key: $ADMIN_API_KEY" http://localhost:8000/status | jq

# Verify error responses
curl http://localhost:8000/stats | jq .error
```

---

## 📖 Additional Resources

- **Complete Guide**: [doc/ADMIN_ENDPOINTS_SECURITY.md](doc/ADMIN_ENDPOINTS_SECURITY.md)
- **API Documentation**: [doc/API_DOCUMENTATION.md](doc/API_DOCUMENTATION.md)
- **Security Features**: [doc/SECURITY-FEATURES.md](doc/SECURITY-FEATURES.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## 💡 FAQ

### Q: Can I disable authentication for development?
**A:** Yes, set `ENABLE_PUBLIC_STATS=true` and `ENABLE_PUBLIC_STATUS=true`. NOT recommended for production.

### Q: What if I forget my API key?
**A:** Generate a new key and update environment variables. Old key will be invalidated immediately.

### Q: Can I use different keys for different teams?
**A:** Current implementation supports one global key. For multi-key support, consider implementing an API key database.

### Q: Does this affect the `/health` endpoint?
**A:** No, `/health` remains public for basic monitoring and health checks.

### Q: What about the `/predict` endpoint?
**A:** No change. `/predict` remains public with existing rate limiting.

### Q: Can I monitor the API without admin access?
**A:** Yes, use the public `/health` endpoint for basic status monitoring.

### Q: How do I rotate API keys?
**A:** Generate new key → Update environment → Restart service → Old key immediately invalid.

### Q: What happens if ADMIN_API_KEY is not set?
**A:** Endpoints return 503 (Service Unavailable) until key is configured.

---

## 🆘 Support

**Issues?** Open a GitHub issue with:
- Error messages
- Environment configuration (redact sensitive keys)
- Steps to reproduce
- Expected vs actual behavior

**Security Concerns?** Contact: security@yourcompany.com

---

## ✅ Checklist

Before deploying v3.5.1, ensure:

- [ ] Admin API key generated
- [ ] Environment variables configured
- [ ] Monitoring scripts updated
- [ ] CI/CD pipelines updated
- [ ] Dashboard tools reconfigured
- [ ] Third-party integrations notified
- [ ] Documentation reviewed
- [ ] Testing completed
- [ ] Team trained on new authentication
- [ ] Rollback plan documented

---

**Version**: 3.5.1  
**Date**: November 9, 2025  
**Author**: Security Team  
**Status**: Production Ready ✅
