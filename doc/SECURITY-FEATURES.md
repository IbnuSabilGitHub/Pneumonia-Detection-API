# Pneumonia Detection API - Security Documentation v3.1.0

## Advanced Security Features for Production & Learning

This API implements **enterprise-grade security features** with advanced protection mechanisms. The following are the comprehensive security features implemented in v3.1.0:

## 🛡️ Advanced Rate Limiting & Attack Protection

### 1. IP Switching Attack Protection
- **Advanced Detection**: Multi-layered fingerprinting system
- **Request Fingerprinting**: User-agent, language, encoding analysis
- **Behavioral Analysis**: Pattern detection for automated attacks
- **Global Attack Scoring**: Real-time threat assessment
- **Automatic Blocking**: Intelligent IP and fingerprint blocking
- **Storage Backend**: Redis (production) / Memory (development)

### 2. Enhanced Rate Limiting
- **Multi-tier Protection**: IP-based and fingerprint-based limits
- **Endpoint Protection**: All prediction endpoints protected
- **Intelligent Limits**: 
  - 5 requests per minute per IP
  - 3 requests per fingerprint pattern
  - Temporary blocks for 5-10 minutes
- **Fallback Mechanisms**: Graceful degradation when systems unavailable
- **Real-time Monitoring**: Live security status and metrics

### 3. Request Fingerprinting
- **User-Agent Analysis**: Browser/client identification
- **Language Fingerprinting**: Accept-Language header analysis  
- **Connection Patterns**: TCP connection behavior analysis
- **Temporal Analysis**: Request timing pattern detection
- **Hash-based Tracking**: SHA-256 fingerprint generation

## 🔐 Enhanced Security Middleware

### 4. Modular Security Architecture
- **SecurityMiddleware**: Refactored with clean separation of concerns
- **Helper Methods**: Modular rate limiting checks
- **Error Handling**: Improved exception propagation
- **Fallback Systems**: Graceful handling of component failures
- **Security Headers**: Comprehensive HTTP security headers

### 5. File Validation & Content Security
- **Size Limit**: Maximum 10MB per file
- **File Types**: Only JPG, JPEG, PNG allowed
- **Content Validation**: Advanced X-ray image characteristics validation
- **Hash-based Integrity**: SHA-256 file content verification
- **Medical Image Validation**: AI-powered medical relevance checking

### 6. Advanced Duplicate Detection
- **Multi-layer Hashing**: SHA-256 file content + metadata hashing
- **Intelligent Caching**: Redis-based distributed cache (production)
- **Time-based Expiry**: 5-minute cache duration with smart cleanup
- **Cross-session Prevention**: Global duplicate tracking

### 7. Production Security Headers
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY  
X-XSS-Protection: 1; mode=block
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 3
X-RateLimit-Reset: 1635789600
Retry-After: 60
```

## ⚙️ Configuration & Environment

### Advanced Security Configuration

```bash
# Application Settings
APP_VERSION=3.1.0
DEBUG=false
STORAGE_BACKEND=redis  # redis (production) / memory (development)

# Advanced Rate Limiting
ADVANCED_RATE_LIMITING_ENABLED=true
MAX_REQUESTS_PER_IP=10
MAX_FINGERPRINT_REQUESTS=3
IP_BLOCK_DURATION=600  # 10 minutes
FINGERPRINT_BLOCK_DURATION=300  # 5 minutes
ATTACK_SCORE_THRESHOLD=0.7
RECENT_IPS_WINDOW=30  # seconds

# Redis Configuration (Production)
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_secure_password
REDIS_DB=0
REDIS_MAX_CONNECTIONS=20

# Security Headers
TRUSTED_HOSTS=*.railway.app,localhost,127.0.0.1
CORS_ORIGINS=*  # Configure for your frontend domains

# File Upload Security
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png
```

## 🚀 Advanced API Usage

### Security Status Monitoring
```bash
# Get comprehensive security status
curl -X GET "http://localhost:8000/security/status"
```

**Enhanced Response:**
```json
{
  "service": "Pneumonia Detection API",
  "version": "3.1.0",
  "security_status": "active",
  "timestamp": "2025-08-25T18:45:00.000Z",
  "rate_limiter": {
    "storage_backend": "redis",
    "storage_initialized": true,
    "active_ips": 5,
    "blocked_fingerprints": 2,
    "global_attack_score": 0.3,
    "recent_ips_count": 3
  },
  "advanced_features": [
    "IP Switching Attack Protection",
    "Request Fingerprinting",
    "Behavioral Analysis", 
    "Global Attack Scoring",
    "Redis Distributed Storage",
    "Enhanced Security Headers"
  ]
}
```

### Security Statistics Endpoint
```bash
# Get detailed security statistics  
curl -X GET "http://localhost:8000/security/stats"
```

**Response:**
```json
{
  "current_active_ips": 12,
  "blocked_fingerprints": 3,
  "global_attack_score": 0.45,
  "total_requests_blocked": 23,
  "fingerprint_analysis": {
    "unique_fingerprints": 8,
    "suspicious_patterns": 2,
    "behavioral_anomalies": 1
  },
  "storage_info": {
    "backend": "redis",
    "connection_status": "healthy",
    "total_keys": 156,
    "memory_usage": "2.3MB"
  }
}
```

## 🛡️ Advanced Error Responses & Security Events

### Enhanced Rate Limit Response (429)
```json
{
  "error": "Rate limit exceeded",
  "message": "IP switching attack detected - fingerprint blocked",
  "client_ip": "192.168.1.100", 
  "endpoint": "POST /pneumonia/predict",
  "timestamp": 1635789600,
  "details": {
    "attack_score": 0.85,
    "fingerprint_blocked": true,
    "block_duration": 600,
    "reason": "Suspicious request pattern detected"
  }
}
```

### IP Switching Attack Detection (429)
```json
{
  "error": "Security violation detected",
  "message": "IP switching attack pattern identified",
  "attack_details": {
    "attack_score": 0.92,
    "ip_switches": 5,
    "time_window": "30 seconds", 
    "fingerprint_consistency": false,
    "behavioral_anomalies": ["rapid_ip_switching", "inconsistent_headers"]
  },
  "action_taken": "Temporary fingerprint block (10 minutes)"
}
```

### Storage Backend Failure (503)
```json
{
  "error": "Service temporarily unavailable",
  "message": "Rate limiting storage backend unavailable",
  "fallback_mode": true,
  "details": {
    "backend": "redis",
    "error": "Connection timeout",
    "fallback_behavior": "Memory-based rate limiting active"
  }
}
```

## 🚀 Production Deployment Security

### Railway/Cloud Environment Variables
Set these in your deployment platform:

```bash
# Core Security
APP_VERSION=3.1.0
DEBUG=false
STORAGE_BACKEND=redis

# Redis Configuration (Production)
REDIS_URL=redis://your-redis-instance:6379
REDIS_PASSWORD=your_secure_redis_password
REDIS_DB=0
REDIS_MAX_CONNECTIONS=20

# Advanced Rate Limiting
ADVANCED_RATE_LIMITING_ENABLED=true
MAX_REQUESTS_PER_IP=10
MAX_FINGERPRINT_REQUESTS=3
ATTACK_SCORE_THRESHOLD=0.7

# Security Headers & CORS
TRUSTED_HOSTS=your-domain.com,*.your-domain.com
CORS_ORIGINS=https://your-frontend.com,https://your-app.com
```

### Redis Production Setup
For high-security production environments:

1. **Dedicated Redis Instance**: Use managed Redis service
2. **Authentication**: Strong password and SSL/TLS
3. **Network Security**: VPC/firewall restrictions
4. **Monitoring**: Redis performance and security monitoring
5. **Backup**: Regular automated backups

### Container Security (Docker)
```dockerfile
# Enhanced security in production container
FROM python:3.11-slim

# Security updates
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Non-root user
RUN useradd -m -u 1000 pneumonia-api
USER pneumonia-api

# Security environment
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application with security context
WORKDIR /app
COPY --chown=pneumonia-api:pneumonia-api . .
```

## 🔍 Advanced Security Best Practices

### 1. Infrastructure Security
- **HTTPS Everywhere**: Enforce TLS 1.3 in production
- **API Gateway**: Use reverse proxy (Nginx/Cloudflare) for additional protection
- **Network Segmentation**: Isolate Redis and application servers
- **Regular Updates**: Automated security patches and dependency updates
- **Container Security**: Minimal containers with security scanning

### 2. Monitoring & Threat Detection
- **Real-time Monitoring**: Track attack patterns and security events
- **Alerting Systems**: Automated alerts for security violations
- **Log Analysis**: Centralized logging with security event correlation
- **Metrics Collection**: Security KPIs and attack success rates
- **Incident Response**: Automated blocking and manual investigation workflows

### 3. Advanced Configuration
```python
# Production security settings
ADVANCED_SECURITY_CONFIG = {
    "rate_limiting": {
        "storage_backend": "redis",
        "distributed_locks": True,
        "attack_detection": True,
        "behavioral_analysis": True
    },
    "fingerprinting": {
        "enabled": True,
        "hash_algorithm": "sha256",
        "include_headers": ["user-agent", "accept-language", "accept-encoding"],
        "temporal_analysis": True
    },
    "blocking": {
        "ip_block_duration": 600,  # 10 minutes
        "fingerprint_block_duration": 300,  # 5 minutes
        "progressive_penalties": True,
        "whitelist_enabled": True
    }
}
```

## 📊 Security Monitoring & Metrics

### Key Security Metrics to Monitor:
- **Attack Detection Rate**: % of automated attacks identified
- **False Positive Rate**: Legitimate requests incorrectly blocked
- **Response Time Impact**: Security overhead on API performance
- **Storage Performance**: Redis connection health and latency
- **Block Effectiveness**: Reduced malicious traffic after blocking

### Alert Thresholds:
- **High Attack Score**: Global score > 0.8 for 5+ minutes
- **Storage Failures**: Redis connection failures > 3 in 1 minute  
- **Unusual Traffic**: 10x normal request volume from single IP
- **Fingerprint Anomalies**: 5+ new suspicious fingerprints in 1 minute

### Dashboard Metrics:
```json
{
  "security_health": {
    "overall_status": "healthy",
    "attack_detection": "active", 
    "storage_backend": "redis_healthy",
    "blocked_requests_last_hour": 12,
    "unique_attackers_blocked": 3,
    "avg_response_time_ms": 45
  }
}
```

## 🚨 Incident Response Procedures

### 1. Automated Response
- **Immediate Blocking**: High-confidence attacks blocked automatically
- **Progressive Penalties**: Repeat offenders get longer blocks
- **Whitelist Override**: Trusted IPs bypass certain restrictions
- **Fallback Mode**: Graceful degradation during system issues

### 2. Manual Investigation
- **Security Logs**: Detailed forensic data for manual review
- **Pattern Analysis**: Historical attack data for trend identification  
- **IP Reputation**: Integration with threat intelligence feeds
- **Custom Rules**: Manual rule creation for specific threats

### 3. Recovery Procedures
- **Block List Management**: Manual unblocking of false positives
- **System Health Checks**: Automated verification after incidents
- **Performance Monitoring**: Ensure security doesn't impact legitimate users
- **Documentation**: Incident logging and lessons learned

---

## ⚖️ Security Compliance & Considerations

### Medical Data Security
- **No Patient Data**: API processes images only, no personal information stored
- **Temporary Processing**: Images processed in memory, not persisted
- **Audit Logging**: All prediction requests logged for security auditing
- **Disclaimer Enforcement**: Clear medical disclaimer in all responses

### Privacy Protection
- **IP Anonymization**: Option to hash/anonymize IP addresses in logs
- **Data Minimization**: Only collect necessary security metadata
- **Retention Policies**: Automated cleanup of old security logs
- **GDPR Compliance**: European data protection considerations

---

**Security Status: 🛡️ PRODUCTION-READY with Advanced Protection**

*Pneumonia Detection API v3.1.0 - Enterprise-grade security with IP switching attack protection and distributed rate limiting*
