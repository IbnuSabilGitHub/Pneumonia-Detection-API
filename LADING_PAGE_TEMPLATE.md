# Pneumonia Detection API
AI-powered chest X-ray analysis with production-grade security

[Primary CTA: Try the API]({{cta_try_url}})   [Secondary CTA: View Docs]({{cta_docs_url}})

—

Medical disclaimer: This API is for educational and research purposes only and must not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals.

—


Section: Hero
- Headline (short): Detect pneumonia from chest X-rays in seconds
- Headline (alt): Fast, reliable AI analysis for chest X-ray images
- Subheadline: Upload an image, get a prediction with confidence scores—secured by advanced, multi-layer protection.
- Primary CTA: Try the API → {{cta_try_url}}
- Secondary CTA: View Docs → {{cta_docs_url}}
- Tertiary CTA (optional): Check Status → {{cta_status_url}}
- Trust badges (text-only placeholders):
  - Built with FastAPI
  - Powered by ONNX
  - Advanced Rate Limiting
  - Production Ready on Railway


Section: Who is this for?
- Developers building medical AI demos and research tools
- Educators teaching medical imaging and ML concepts
- Teams prototyping secure, production-style AI inference APIs


Section: Key Benefits
- Accurate AI predictions with confidence scores
- Two model options: Standard CNN and EfficientNet-B0
- Strong security: fingerprinting, IP switch detection, behavioral analysis
- Simple deployment: in-memory storage by default; Redis optional
- Clear documentation, examples, and monitoring endpoints


Section: How it works (3 steps)
1) Upload a chest X-ray (JPG/JPEG/PNG, max 10MB)
2) Choose a model (standard or efficientnet_b0)
3) Receive a prediction with confidence and a safety recommendation

API base URL: {{base_url}}
- Health: {{base_url}}/health
- Predict: {{base_url}}/pneumonia/predict
- Model info: {{base_url}}/pneumonia/model/info
- Security status: {{base_url}}/security/status


Section: Live request examples
- cURL (Standard model)
  ```
  curl -X POST "{{base_url}}/pneumonia/predict" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@chest_xray.jpg"
  ```
- cURL (EfficientNet-B0)
  ```
  curl -X POST "{{base_url}}/pneumonia/predict?model=efficientnet_b0" \
       -H "Content-Type: multipart/form-data" \
       -F "file=@chest_xray.jpg"
  ```

Sample response
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


Section: Models overview
- Standard (Baseline CNN)
  - ~200ms inference, ~91% accuracy, ~15MB
  - Best for high-volume or quick screening
- EfficientNet-B0
  - ~300ms inference, ~95% accuracy, ~20MB
  - Best for higher-accuracy use cases

Model selection guideline:
- Use Standard for: faster results and batch workloads
- Use EfficientNet-B0 for: critical analysis and higher accuracy


Section: Security-first by design
- Multi-layer Advanced Rate Limiting
  - IP-based limits: default 5 requests/min per IP
  - Fingerprint limits: default 2 requests/min per browser pattern
  - Temporary blocks on suspicious activity
- IP Switching Attack Detection
  - Detects same fingerprint across multiple IPs (rotating proxies/VPNs)
  - Flags distributed bursts from many IPs
- Behavioral Analysis
  - Detects low timing variance (bot-like intervals)
  - Duplicate file detection across IPs
- Global Attack Scoring
  - Dynamic risk scoring adjusts limits under attack

Security endpoints:
- {{base_url}}/security/status
- {{base_url}}/security/stats


Section: Rate limiting at a glance
- Default limits
  - Predict endpoints: 5 requests per minute per IP
  - Fingerprint limit: 2 per minute (strict)
- Client guidance
  - Implement exponential backoff (e.g., 60s)
  - Watch X-RateLimit-* and Retry-After headers
  - Cache results when appropriate


Section: Performance & reliability
- In-memory storage: ~0.1ms security overhead per request
- Total API overhead: ~2% response time increase
- Startup: < 1 second (no external services)
- Memory usage: ~20–30MB (no Redis)
- Optional Redis for multi-instance or persistence


Section: Monitoring & status
- Health: {{base_url}}/health
- Security: {{base_url}}/security/status
- Docs: {{base_url}}/docs
- ReDoc: {{base_url}}/redoc

Suggested alert thresholds:
- Attack Score: warn > 0.7, critical > 0.9
- Blocked rate: warn > 20%, critical > 50%
- Unique IPs/min: warn > 10, critical > 20
- Response time: warn > 100ms, critical > 500ms


Section: Deployment options
- Railway (recommended)
  - railway up
- Docker
  - docker build -t pneumonia-api:{{version}} .
  - docker run -p 8000:8000 -e APP_VERSION={{version}} pneumonia-api:{{version}}
- Local development
  - pip install -r requirements.txt
  - python main.py


Section: Configuration (common envs)
```
APP_VERSION={{version}}
STORAGE_BACKEND=memory
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW=60
MAX_FINGERPRINT_REQUESTS=2
ATTACK_SCORE_THRESHOLD=0.6
IP_SWITCH_THRESHOLD=3
MEMORY_MAX_SIZE=10000
```

High-security preset
```
RATE_LIMIT_REQUESTS=3
MAX_FINGERPRINT_REQUESTS=1
ATTACK_SCORE_THRESHOLD=0.4
IP_SWITCH_THRESHOLD=2
```


Section: FAQ
- Is this a medical device?
  - No. For education/research only. Not for clinical use.
- Do you store my images?
  - Images are processed in memory and not persisted by default.
- What file types are supported?
  - JPG, JPEG, PNG. Max size 10MB.
- Why did I get a 429 error?
  - You hit a rate limit or triggered security rules. Wait and retry with backoff.
- When should I use EfficientNet-B0?
  - When accuracy is more important than speed.

If you’re scaling beyond a single instance:
- Consider Redis for distributed rate limiting and persistence.
- Set STORAGE_BACKEND=redis and configure REDIS_URL.


Section: CTA (footer)
- Try the API → {{cta_try_url}}
- View Documentation → {{cta_docs_url}}
- Check Security Status → {{cta_status_url}}

—

Legal & Privacy
- No PHI/PII should be uploaded.
- Images are validated for medical relevance but not stored by default.
- Review your compliance requirements before production use.

—


Content variables (replace in your frontend):
- {{base_url}}: e.g., http://localhost:8000
- {{version}}: e.g., 3.4.2
- {{cta_try_url}}: e.g., {{base_url}}/docs
- {{cta_docs_url}}: e.g., {{base_url}}/docs
- {{cta_status_url}}: e.g., {{base_url}}/security/status


Copy variants (for A/B testing)
- Headline A: Detect pneumonia from chest X-rays in seconds
- Headline B: AI-powered chest X-ray analysis—fast, accurate, secure
- Subheadline A: Upload an image, choose a model, get a prediction with confidence scores.
- Subheadline B: Enterprise-grade protection meets developer-friendly AI inference.

SEO suggestions
- Title: Pneumonia Detection API | AI Chest X-ray Analysis (FastAPI + ONNX)
- Meta description: Upload chest X-rays and get AI predictions with confidence scores. Secure by design with multi-layer protection. FastAPI + ONNX. In-memory by default; Redis optional.
- Keywords: pneumonia detection API, chest x-ray AI, fastapi onnx, medical imaging inference, rate limiting, ip switching detection