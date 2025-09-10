# 🏥 Pneumonia Detection API

Advanced AI powered medical imaging analysis service for detecting pneumonia from chest X-ray images. Built with FastAPI, powered by ONNX, and secured with a multi layer, production grade protection system.


---

## ⚠️ Medical Disclaimer

This API is for educational and research purposes only. Predictions must **NEVER** be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.

---

## 📚 Table of Contents

1. Overview
2. Key Features
3. Architecture
4. Quick Start
5. API Endpoints
6. Security & Rate Limiting
7. Configuration
8. Deployment
9. Usage Examples
10. Monitoring & Alerting
11. Testing & Quality
12. Migration: Redis → In-Memory
13. Performance & Benchmarks
14. Troubleshooting
15. References
16. Changelog (Summary)

---

## 🌟 Overview

The Pneumonia Detection API provides AI inference for chest X-ray images using deep learning (Standard CNN, EfficientNet-B0). It includes enterprise grade security with advanced rate limiting, request fingerprinting, IP switching detection, behavioral analysis, and global attack scoring all optimized for single-instance deployments (e.g., Railway) using in-memory storage by default.

---

## 🚀 Key Features

- AI-Powered Detection
  - Models: Standard CNN and EfficientNet-B0
  - Confidence scoring and class probabilities
  - Medical recommendations included in responses

- Security & Protection
  - Multi-layer advanced rate limiting
  - Request fingerprinting
  - IP switching attack detection
  - Behavioral analysis and file-duplication detection
  - Global attack scoring with dynamic limits
  - Comprehensive security status endpoints

- Operational Excellence
  - Modular, maintainable architecture
  - In-memory storage default (Redis optional)
  - Rich API documentation and examples
  - Optimized for Railway and Docker

---

## 🏗️ Architecture

Clean, modular architecture with clear separation of concerns and production-grade patterns.

Project structure:
```
pneumonia-detection-api/
├── app/
│   ├── main.py                         # FastAPI application factory
│   ├── api/                            # API routes
│   │   ├── health.py
│   │   ├── prediction.py
│   │   └── security.py
│   ├── services/                       # Business logic
│   │   └── prediction.py
│   ├── models/                         # Pydantic schemas
│   ├── core/                           # Core & infrastructure
│   │   ├── advanced_rate_limiting.py
│   │   ├── memory_storage.py
│   │   ├── redis_storage.py            # Optional
│   │   ├── storage_factory.py
│   │   ├── storage_backends.py
│   │   ├── settings.py
│   │   └── logger.py
│   ├── middleware/
│   │   └── security.py
│   └── utils/
│       ├── exceptions.py
│       ├── security.py
│       └── validation.py
├── models/                             # ONNX model files
├── Dockerfile
├── requirements.txt
└── README.md
```

Recent improvements:
- Refactored middleware and services for maintainability
- Strong error handling and graceful fallbacks
- In-memory default storage with optional Redis support
- Enhanced API docs and schema clarity

---

## ⚡ Quick Start

Clone, install, and run:

```bash
git clone https://github.com/IbnuSabilGitHub/Pneumonia-Detection-API.git
cd Pneumonia-Detection-API

python -m venv .venv
. .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt

python main.py  # or: uvicorn app.main:app --reload
```

Base URL
```
http://localhost:8000
```

Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

Basic Prediction
```bash
curl -X POST "http://localhost:8000/pneumonia/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"
```

---

## 📡 API Endpoints

Health & Monitoring
- GET / or GET /health
  - Returns: { status: healthy|partial|unhealthy, model_loaded, version, uptime }

Security
- GET /security/status
  - Real-time protection status and metrics
- GET /security/stats
  - Detailed analytics and interpretation

Pneumonia Detection
- POST /pneumonia/predict
  - Params: file (JPG/JPEG/PNG, max 10MB), model=standard|efficientnet_b0
  - Returns: prediction, confidence, probabilities, recommendation, model info, disclaimer
- GET /pneumonia/model/info
  - Params: model=standard|efficientnet_b0
  - Returns: model metadata (architecture, input shape, performance)

Example Response (Predict):
```json
{
  "prediction": "NORMAL",
  "confidence": 0.92,
  "probabilities": { "NORMAL": 0.92, "PNEUMONIA": 0.08 },
  "medical_recommendation": "✅ Normal results - maintain regular health checkups",
  "model_version": "v1.0",
  "model_type": "efficientnet_b0",
  "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
}
```

Common Error Responses
- 400 INVALID_FILE_FORMAT, INVALID_IMAGE_CONTENT
- 409 DUPLICATE_FILE
- 413 FILE_TOO_LARGE
- 429 RATE_LIMIT_EXCEEDED
- 503 SERVICE_UNAVAILABLE

---

## 🛡️ Security & Rate Limiting

Why advanced rate limiting?
- Traditional IP-only limits are easy to bypass (VPNs/proxies)
- No browser fingerprint detection
- No behavioral analysis
- No distributed attack detection

Solution: Multi-layer protection with dynamic scoring
- Layer 1: Fingerprint block check
- Layer 2: IP rate limiting
- Layer 3: Fingerprint rate limiting
- Layer 4: IP switching detection (same fingerprint across many IPs, distributed bursts)
- Layer 5: Behavioral analysis (timing variance, duplicate file across IPs)
- Layer 6: Global attack scoring (request rate, unique IPs, blocked fingerprints) driving stricter temporary limits

Default thresholds (tunable):
```python
WINDOW_SIZE = 60                      # seconds
MAX_REQUESTS_PER_IP = 5               # per minute
MAX_FINGERPRINT_REQUESTS = 2          # per minute
IP_SWITCHING_THRESHOLD = 3            # same fingerprint from 3+ IPs
GLOBAL_ATTACK_THRESHOLD = 0.6
ATTACK_BLOCK_DURATION = 300           # seconds
BOT_BEHAVIOR_VARIANCE = 0.1
```

What’s protected
- POST /pneumonia/predict: full protection
- Security endpoints & health: generally unthrottled or relaxed limits

Security headers (examples)
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 3
X-RateLimit-Reset: 1625097600
Retry-After: 60
```

---

## ⚙️ Configuration

Environment Variables (common)
```
# App
APP_VERSION=3.4.2
DEBUG=false

# Storage (default: memory)
STORAGE_BACKEND=memory   # memory | redis

# In-Memory Storage
MEMORY_MAX_SIZE=10000
MEMORY_CLEANUP_INTERVAL=300
MEMORY_DEFAULT_TTL=3600

# Rate Limiting
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW=60
RATE_LIMIT_BLOCK_DURATION=300

# Advanced Protection
ADVANCED_RATE_LIMITING_ENABLED=true
MAX_FINGERPRINT_REQUESTS=2
ATTACK_SCORE_THRESHOLD=0.6
IP_SWITCH_THRESHOLD=3

# Redis (optional / production)
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_secure_password
REDIS_DB=0
REDIS_MAX_CONNECTIONS=20

# CORS / Hosts
TRUSTED_HOSTS=*.railway.app,localhost,127.0.0.1
CORS_ORIGINS=*
```

High-Security preset
```
RATE_LIMIT_REQUESTS=3
MAX_FINGERPRINT_REQUESTS=1
ATTACK_SCORE_THRESHOLD=0.4
IP_SWITCH_THRESHOLD=2
```

---

## 🌐 Deployment

Option 1: Railway (Recommended)
```bash
railway login
railway up
railway status
```

Option 2: Docker
```bash
docker build -t pneumonia-api:v3.4.2 .
docker run -p 8000:8000 -e APP_VERSION=3.4.2 pneumonia-api:v3.4.2
```

Option 3: Heroku
```bash
heroku login
heroku create your-app-name
git push heroku main
```

Option 4: Local Development
```bash
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
# source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

Post-deployment verification
- GET /health → status, model_loaded, version, uptime
- GET /security/status → active, metrics
- GET /docs → interactive docs
- GET /pneumonia/model/info → model stats

---

## 🔧 Usage Examples

cURL
```bash
# Health
curl -X GET "http://localhost:8000/health"

# Predict (standard)
curl -X POST "http://localhost:8000/pneumonia/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"

# Predict (EfficientNet-B0)
curl -X POST "http://localhost:8000/pneumonia/predict?model=efficientnet_b0" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"

# Model info
curl -X GET "http://localhost:8000/pneumonia/model/info?model=standard"

# Security status
curl -X GET "http://localhost:8000/security/status"
```

Python (requests)
```python
import requests
with open("chest_xray.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/pneumonia/predict",
        files={"file": ("chest_xray.jpg", f, "image/jpeg")},
        params={"model": "efficientnet_b0"},
        timeout=30
    )
print(r.status_code, r.json())
```

Node.js (axios)
```javascript
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

(async () => {
  const form = new FormData();
  form.append("file", fs.createReadStream("chest_xray.jpg"));
  const res = await axios.post(
    "http://localhost:8000/pneumonia/predict?model=standard",
    form,
    { headers: form.getHeaders() }
  );
  console.log(res.data);
})();
```

More examples: Python async, Browser fetch, Java, C#, PHP available in USAGE_EXAMPLES.

---

## 📊 Monitoring & Alerting

Key metrics
- Requests per minute, success rate, response times
- Global attack score, blocked requests, unique IP count
- Memory/CPU usage, cache hit rate

Suggested alert thresholds
- Attack Score: warn > 0.7, critical > 0.9
- Blocked Rate: warn > 20%, critical > 50%
- Unique IPs/min: warn > 10, critical > 20
- Response Time: warn > 100ms, critical > 500ms

Security status sample
```json
{
  "service": "Pneumonia Detection API",
  "security_status": "active",
  "advanced_protection": {
    "global_attack_score": 0.15,
    "requests_per_minute": 23,
    "recent_unique_ips": 8,
    "blocked_fingerprints": 2,
    "storage_type": "memory"
  }
}
```

---

## 🧪 Testing & Quality

Available test scripts (examples)
- test_rate_limiting.py
- test_concurrent_rate_limiting.py
- test_file_upload_rate_limiting.py
- test_rate_limit_recovery.py
- test_ip_switching_attack.py
- run_all_rate_limit_tests.py

Advanced protection expected results (sample)
- Overall Security Score: ~83.3/100 (Excellent)
- Fingerprint Detection: 100%
- IP Switching Protection: ~58.3%
- Basic Rate Limiting: Working

Final verification checklist
- Application starts successfully
- Rate limiter initializes properly (in-memory)
- All endpoints functional (/health, /docs, /security/status)
- Performance optimized (low overhead)
- Error handling robust

---

## 🔄 Migration: Redis → In-Memory

Why migrate?
- Simpler deployment (single service)
- Lower memory and latency
- Perfect for Railway and education
- Zero external dependencies

What changed
- Default storage_backend: redis → memory
- Optional Redis dependency
- Storage factory supports graceful fallback
- Updated docs, deployment, and tests

Performance improvements
- Memory: ~80MB → ~30MB (−62.5%)
- Response overhead: +2–5ms → +0.1ms (−95%)
- Startup: 2–5s → <1s (−80%)
- Dependencies: 15 → 10 (−33%)

When to use Redis again
- >1000 concurrent users
- Multi-instance behind a load balancer
- Need persistence across restarts
- Advanced analytics and distributed rate limiting

How to re-enable Redis
```
# requirements.txt
redis>=5.0.0

# env
STORAGE_BACKEND=redis
REDIS_URL=redis://your-redis:6379
REDIS_PASSWORD=****
```

---

## 📈 Performance & Benchmarks

In-memory backend
- Total memory: ~20–30MB
- Overhead: ~0.1ms/request
- Response time increase: ~2%
- Startup time: <1s

Advanced detection accuracy
- Overall: ~83.3/100
- Layers: IP switching, fingerprinting, behavior analysis, global scoring

---

## 🛠️ Troubleshooting

Common issues
- Invalid file: Ensure JPG/JPEG/PNG; max 10MB; valid X-ray content
- Rate limit 429: Implement backoff; space out requests; monitor headers
- Slow responses: Reduce image size; choose Standard model; check CPU/memory
- Model not loading: Verify ONNX files; ONNX Runtime installed; check logs

Debug tips
```bash
curl http://localhost:8000/health | jq
curl http://localhost:8000/security/status | jq
tail -f logs/security.log | grep "BLOCKED\|ATTACK"
```

False positives/negatives
- Tune thresholds (increase/decrease limits)
- Consider whitelisting legitimate patterns
- Review blocked fingerprints weekly; analyze attack patterns monthly

---

## 🔗 References

- OWASP Rate Limiting Guide: https://owasp.org/www-community/controls/Blocking_Brute_Force_Attacks
- RFC 6585 - HTTP 429: https://tools.ietf.org/html/rfc6585
- Cloudflare Rate Limiting: https://developers.cloudflare.com/fundamentals/api/get-started/requests-per-minute
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework

Built with FastAPI | Powered by ONNX | Secured by Design