# 🛡️ Advanced Rate Limiting & IP Switching Attack Protection

## 📚 Overview
This documentation explains the implementation of **Advanced Rate Limiting** with protection against **IP Switching Attacks** on the Pneumonia Detection API. The system uses a multi-layer security approach to protect the API from various types of attacks.

## 🎯 Problems Solved

### ❌ Weaknesses of Traditional Rate Limiting:
- **IP-based only**: Easy to bypass with VPN/proxy
- **No fingerprinting**: Does not detect attack patterns
- **No behavioral analysis**: Does not recognize bot behavior
- **No distributed attack detection**: Does not detect coordinated attacks

### ✅ Advanced Rate Limiting Solution:
- **Multi-layer protection**: 5 different security layers
- **Request fingerprinting**: Browser signature detection
- **IP switching detection**: Detects suspicious IP changes
- **Behavioral analysis**: Analyzes request patterns
- **Global attack scoring**: Dynamic threat assessment

Translated with DeepL.com (free version)

## 🏗️ Arsitektur Sistem

```mermaid
graph TB
    A[Client Request] --> B[Security Middleware]
    B --> C{Advanced Rate Limiter}
    
    C --> D[Layer 1: Fingerprint Block Check]
    D --> |Blocked| Z[Return 429]
    D --> |Pass| E[Layer 2: IP Rate Limiting]
    
    E --> |Exceeded| Z
    E --> |Pass| F[Layer 3: Fingerprint Rate Limiting]
    
    F --> |Exceeded| G[Block Fingerprint]
    G --> Z
    F --> |Pass| H[Layer 4: IP Switching Detection]
    
    H --> |Detected| I[Block Fingerprint]
    I --> Z
    H --> |Pass| J[Layer 5: Behavioral Analysis]
    
    J --> |Anomaly| K[Block Fingerprint]
    K --> Z
    J --> |Pass| L[Layer 6: Global Attack Score]
    
    L --> |High Score| M[Reduced Limits]
    M --> |Exceeded| Z
    L --> |Pass| N[Allow Request]
    M --> |Pass| N
    
    N --> O[Process Request]
    O --> P[Return Response]
    
    style A fill:#e1f5fe
    style C fill:#fff3e0
    style Z fill:#ffebee
    style N fill:#e8f5e8
    style O fill:#e8f5e8
    style P fill:#e8f5e8
```

## 🔄 Flow Diagram Lengkap

```mermaid
flowchart TD
    Start([Client Makes Request]) --> MW[Security Middleware]
    
    MW --> ExtractIP[Extract Client IP]
    MW --> CreateFP[Create Request Fingerprint]
    
    ExtractIP --> GetHeaders[Get Headers: X-Forwarded-For, X-Real-IP]
    CreateFP --> HashUA[Hash User-Agent + Headers]
    
    GetHeaders --> IPCheck{Layer 1: Check Blocked Fingerprints}
    HashUA --> IPCheck
    
    IPCheck --> |Blocked| Return429[Return HTTP 429]
    IPCheck --> |Not Blocked| IPRateLimit{Layer 2: IP Rate Limiting}
    
    IPRateLimit --> CheckIPCount[Check IP Request Count in Window]
    CheckIPCount --> |> 5 requests| Return429
    CheckIPCount --> |<= 5 requests| FingerprintLimit{Layer 3: Fingerprint Rate Limiting}
    
    FingerprintLimit --> CheckFPCount[Check Fingerprint Count]
    CheckFPCount --> |> 2 requests| BlockFP1[Block Fingerprint for 5 min]
    BlockFP1 --> Return429
    CheckFPCount --> |<= 2 requests| IPSwitchDetect{Layer 4: IP Switching Detection}
    
    IPSwitchDetect --> TrackIPs[Track Recent IPs]
    TrackIPs --> CheckSameFP[Same Fingerprint from 3+ IPs?]
    CheckSameFP --> |Yes| BlockFP2[Block Fingerprint]
    BlockFP2 --> Return429
    CheckSameFP --> |No| CheckDistributed[5+ Unique IPs in 5 min?]
    CheckDistributed --> |Yes| BlockFP3[Block Fingerprint]
    BlockFP3 --> Return429
    CheckDistributed --> |No| BehaviorAnalysis{Layer 5: Behavioral Analysis}
    
    BehaviorAnalysis --> CheckFilePattern[Same File from 3+ IPs?]
    CheckFilePattern --> |Yes| BlockFP4[Block Fingerprint]
    BlockFP4 --> Return429
    CheckFilePattern --> |No| CheckBotPattern[Regular Request Intervals?]
    CheckBotPattern --> |Yes| BlockFP5[Block Fingerprint]
    BlockFP5 --> Return429
    CheckBotPattern --> |No| GlobalScore{Layer 6: Global Attack Score}
    
    GlobalScore --> CalcScore[Calculate Attack Score]
    CalcScore --> CheckScore{Score > 0.6?}
    CheckScore --> |Yes| ReducedLimits[Apply Stricter Limits]
    ReducedLimits --> |> 1 request| Return429
    ReducedLimits --> |<= 1 request| AllowRequest[Allow Request]
    CheckScore --> |No| AllowRequest
    
    AllowRequest --> ProcessRequest[Process Request]
    ProcessRequest --> LogSuccess[Log Successful Request]
    LogSuccess --> ReturnResponse[Return Response]
    
    Return429 --> LogBlock[Log Blocked Request]
    LogBlock --> End([End])
    ReturnResponse --> End
    
    style Start fill:#e3f2fd
    style Return429 fill:#ffebee
    style AllowRequest fill:#e8f5e8
    style ProcessRequest fill:#e8f5e8
    style ReturnResponse fill:#e8f5e8
    style End fill:#f3e5f5
```

## 🔍 Request Fingerprinting Process

```mermaid
sequenceDiagram
    participant Client
    participant Middleware
    participant AdvancedLimiter
    participant FingerprintDB
    participant IPTracker
    
    Client->>Middleware: HTTP Request
    Middleware->>AdvancedLimiter: Extract IP + Headers
    
    Note over AdvancedLimiter: Create Fingerprint
    AdvancedLimiter->>AdvancedLimiter: Hash(User-Agent + Accept + Language + Encoding)
    
    AdvancedLimiter->>FingerprintDB: Check if fingerprint blocked
    FingerprintDB-->>AdvancedLimiter: Block status
    
    alt Fingerprint is blocked
        AdvancedLimiter-->>Middleware: BLOCKED (429)
        Middleware-->>Client: HTTP 429 + reason
    else Fingerprint not blocked
        AdvancedLimiter->>IPTracker: Track IP + Fingerprint
        IPTracker->>IPTracker: Analyze patterns
        
        alt Suspicious pattern detected
            IPTracker->>FingerprintDB: Block fingerprint
            AdvancedLimiter-->>Middleware: BLOCKED (429)
            Middleware-->>Client: HTTP 429 + reason
        else Normal pattern
            AdvancedLimiter-->>Middleware: ALLOWED
            Middleware->>Middleware: Process request
            Middleware-->>Client: HTTP 200 + response
        end
    end
```

## 🎛️ Configuration & Tuning

### 📊 Default Settings

```python
# Rate Limiting Configuration
WINDOW_SIZE = 60  # 1 minute
MAX_REQUESTS_PER_IP = 10
MAX_FINGERPRINT_REQUESTS = 3  # Strict fingerprint limit
SUSPICIOUS_IP_CHANGES_THRESHOLD = 5  # IPs in short time
ATTACK_BLOCK_DURATION = 300  # 5 minutes

# Attack Detection Thresholds
IP_SWITCHING_THRESHOLD = 3  # Same fingerprint from 3+ IPs
GLOBAL_ATTACK_THRESHOLD = 0.8  # Attack score threshold
BOT_BEHAVIOR_VARIANCE = 0.1  # Request timing variance
```

### ⚙️ Tuning Guidelines

```mermaid
graph LR
    A[Traffic Pattern] --> B{Attack Frequency}
    
    B --> |High| C[Strict Settings]
    B --> |Medium| D[Balanced Settings]  
    B --> |Low| E[Relaxed Settings]
    
    C --> C1[max_requests = 3<br/>fingerprint_limit = 1<br/>attack_threshold = 0.4]
    D --> D1[max_requests = 5<br/>fingerprint_limit = 2<br/>attack_threshold = 0.6]
    E --> E1[max_requests = 10<br/>fingerprint_limit = 3<br/>attack_threshold = 0.8]
    
    style C1 fill:#ffebee
    style D1 fill:#fff3e0
    style E1 fill:#e8f5e8
```

## 🔬 Detection Algorithms

### 1. IP Switching Detection

```mermaid
graph TD
    A[New Request] --> B[Extract Fingerprint]
    B --> C[Check Recent IPs]
    C --> D{Same Fingerprint<br/>from 3+ IPs?}
    D --> |Yes| E[🚨 IP Switching Attack]
    D --> |No| F[Check Unique IPs]
    F --> G{5+ Unique IPs<br/>in 5 minutes?}
    G --> |Yes| H[🚨 Distributed Attack]
    G --> |No| I[✅ Normal Traffic]
    
    style E fill:#ffebee
    style H fill:#ffebee
    style I fill:#e8f5e8
```

### 2. Behavioral Analysis

```mermaid
graph TD
    A[Request Pattern] --> B{File Upload?}
    B --> |Yes| C[Check File Hash]
    B --> |No| D[Check Timing Pattern]
    
    C --> E{Same File from<br/>3+ Different IPs?}
    E --> |Yes| F[🚨 Coordinated Attack]
    E --> |No| D
    
    D --> G[Calculate Request Intervals]
    G --> H{Regular Intervals<br/>+ Low Variance?}
    H --> |Yes| I[🚨 Bot Behavior]
    H --> |No| J[✅ Human Behavior]
    
    style F fill:#ffebee
    style I fill:#ffebee
    style J fill:#e8f5e8
```

### 3. Global Attack Scoring

```mermaid
graph TD
    A[Calculate Attack Score] --> B[Request Rate Factor<br/>40% weight]
    A --> C[Unique IPs Factor<br/>40% weight]
    A --> D[Blocked Fingerprints<br/>20% weight]
    
    B --> E{Requests/min > 100?}
    E --> |Yes| F[Factor = 1.0]
    E --> |No| G[Factor = requests/100]
    
    C --> H{Unique IPs > 50?}
    H --> |Yes| I[Factor = 1.0]
    H --> |No| J[Factor = IPs/50]
    
    D --> K{Blocked > 10?}
    K --> |Yes| L[Factor = 1.0]
    K --> |No| M[Factor = blocked/10]
    
    F --> N[Weighted Sum]
    G --> N
    I --> N
    J --> N
    L --> N
    M --> N
    
    N --> O{Score > 0.6?}
    O --> |Yes| P[🚨 High Attack Risk<br/>Apply Strict Limits]
    O --> |No| Q[✅ Normal Risk]
    
    style P fill:#ffebee
    style Q fill:#e8f5e8
```

## 🧪 Testing Scenarios

### Test Coverage Matrix

```mermaid
graph TB
    subgraph "Attack Scenarios"
        A1[Basic Rate Limiting]
        A2[IP Switching Attack]
        A3[Fingerprint Bypass]
        A4[Coordinated Attack]
        A5[Bot Behavior]
        A6[File Upload Abuse]
    end
    
    subgraph "Protection Layers"
        B1[IP Rate Limiting]
        B2[Fingerprint Limiting]
        B3[IP Switch Detection]
        B4[Behavioral Analysis]
        B5[Global Scoring]
    end
    
    A1 --> B1
    A2 --> B3
    A3 --> B2
    A4 --> B5
    A5 --> B4
    A6 --> B4
    
    style A1 fill:#e3f2fd
    style A2 fill:#fff3e0
    style A3 fill:#f3e5f5
    style A4 fill:#e8f5e8
    style A5 fill:#fce4ec
    style A6 fill:#e0f2f1
```

### Test Results Interpretation

```mermaid
graph LR
    A[Test Results] --> B{Protection Effectiveness}
    
    B --> |> 80%| C[🛡️ EXCELLENT<br/>Strong Protection]
    B --> |60-80%| D[👍 GOOD<br/>Decent Protection]
    B --> |< 60%| E[⚠️ WEAK<br/>Needs Improvement]
    
    C --> F[Ready for Production]
    D --> G[Consider Tuning]
    E --> H[Requires Fixes]
    
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#ffebee
```

## 📈 Performance Impact

### Resource Usage Analysis

```mermaid
graph TD
    A[Request Processing] --> B[Memory Usage]
    A --> C[CPU Usage]
    A --> D[Response Time]
    
    B --> B1[IP Tracking: ~1KB per IP]
    B --> B2[Fingerprint Cache: ~2KB per FP]
    B --> B3[Total: ~50MB for 1000 active users]
    
    C --> C1[Fingerprint Creation: ~0.1ms]
    C --> C2[Pattern Analysis: ~0.5ms]
    C --> C3[Total Overhead: ~0.6ms per request]
    
    D --> D1[Normal Request: 1-5ms]
    D --> D2[With Rate Limiting: 1.6-5.6ms]
    D --> D3[Impact: ~10% increase]
    
    style B3 fill:#e8f5e8
    style C3 fill:#e8f5e8
    style D3 fill:#fff3e0
```

## 🔧 Implementation Details

### File Structure

```
app/
├── core/
│   ├── advanced_rate_limiting.py  # Main implementation
│   └── settings.py               # Configuration
├── middleware/
│   └── security.py              # Middleware integration
├── api/
│   └── security.py              # Security endpoints
└── utils/
    └── security.py              # Helper functions
```

### Key Classes

```mermaid
classDiagram
    class AdvancedRateLimiter {
        +ip_requests: Dict
        +fingerprint_requests: Dict
        +blocked_fingerprints: Dict
        +recent_ips: deque
        +attack_score: float
        +is_request_allowed()
        +detect_ip_switching_attack()
        +detect_behavioral_anomalies()
        +calculate_global_attack_score()
        +create_request_fingerprint()
    }
    
    class SecurityMiddleware {
        +dispatch(request, call_next)
        +get_client_ip(request)
    }
    
    class RequestFingerprint {
        +user_agent_hash: str
        +accept_language: str
        +accept_encoding: str
        +connection_type: str
        +timestamp: float
    }
    
    AdvancedRateLimiter --> RequestFingerprint
    SecurityMiddleware --> AdvancedRateLimiter
```

## 🚀 Usage Examples

### 1. Basic Implementation

```python
from app.core.advanced_rate_limiting import advanced_rate_limiter

# Check if request is allowed
is_allowed, reason, details = advanced_rate_limiter.is_request_allowed(
    client_ip="192.168.1.100",
    endpoint="/api/predict",
    request=request,
    file_hash=file_hash
)

if not is_allowed:
    return JSONResponse(
        status_code=429,
        content={"error": reason, "details": details}
    )
```

### 2. Security Status Monitoring

```python
# Get current security status
status = advanced_rate_limiter.get_security_status()
print(f"Attack Score: {status['global_attack_score']}")
print(f"Active IPs: {status['active_ips']}")
print(f"Blocked Fingerprints: {status['blocked_fingerprints']}")
```

### 3. Custom Configuration

```python
# Adjust thresholds for high-security environment
advanced_rate_limiter.max_requests_per_ip = 3
advanced_rate_limiter.max_fingerprint_requests = 1
advanced_rate_limiter.suspicious_ip_changes_threshold = 3
```

## 📊 Monitoring & Alerting

### Key Metrics to Monitor

```mermaid
graph TB
    A[Monitoring Dashboard] --> B[Request Metrics]
    A --> C[Attack Metrics]
    A --> D[Performance Metrics]
    
    B --> B1[Requests per minute]
    B --> B2[Success rate]
    B --> B3[Response times]
    
    C --> C1[Attack score trend]
    C --> C2[Blocked requests]
    C --> C3[Unique IP count]
    
    D --> D1[Memory usage]
    D --> D2[CPU usage]
    D --> D3[Cache hit rate]
    
    style A fill:#e3f2fd
    style B fill:#e8f5e8
    style C fill:#fff3e0
    style D fill:#f3e5f5
```

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Attack Score | > 0.7 | > 0.9 |
| Blocked Rate | > 20% | > 50% |
| Unique IPs/min | > 10 | > 20 |
| Response Time | > 100ms | > 500ms |

## 🎯 Best Practices

### 1. Security Configuration

```mermaid
graph LR
    A[Security Level] --> B{Choose Configuration}
    
    B --> C[High Security<br/>Banking/Finance]
    B --> D[Medium Security<br/>E-commerce]
    B --> E[Low Security<br/>Public API]
    
    C --> C1[Very strict limits<br/>Aggressive blocking<br/>Low thresholds]
    D --> D1[Balanced limits<br/>Moderate blocking<br/>Standard thresholds]
    E --> E1[Relaxed limits<br/>Minimal blocking<br/>High thresholds]
```

### 2. Deployment Considerations

- **Load Balancer**: Ensure proper IP forwarding headers
- **CDN**: Configure to pass through client IPs
- **Monitoring**: Set up alerts for attack patterns
- **Logging**: Enable detailed security logging
- **Backup**: Implement fallback rate limiting

### 3. Maintenance Tasks

- **Weekly**: Review blocked fingerprints
- **Monthly**: Analyze attack patterns
- **Quarterly**: Tune detection thresholds
- **Annually**: Security audit and updates

## 🔍 Troubleshooting

### Common Issues

```mermaid
graph TD
    A[Issue] --> B{Type of Problem}
    
    B --> C[False Positives]
    B --> D[False Negatives]
    B --> E[Performance Issues]
    
    C --> C1[Reduce sensitivity<br/>Increase thresholds<br/>Whitelist legitimate users]
    
    D --> D1[Increase sensitivity<br/>Lower thresholds<br/>Add more detection layers]
    
    E --> E1[Optimize algorithms<br/>Reduce cache size<br/>Add indexing]
    
    style C1 fill:#fff3e0
    style D1 fill:#ffebee
    style E1 fill:#e8f5e8
```

## 📚 References

- [OWASP Rate Limiting Guide](https://owasp.org/www-community/controls/Blocking_Brute_Force_Attacks)
- [RFC 6585 - HTTP Status Code 429](https://tools.ietf.org/html/rfc6585)
- [CloudFlare Rate Limiting](https://developers.cloudflare.com/fundamentals/api/get-started/requests-per-minute/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

**🛡️ Advanced Rate Limiting Implementation**  
*Protecting APIs from sophisticated attacks with multi-layer security*

**Version**: 3.0.0  
**Last Updated**: August 23, 2025  
**Author**: IbnuSabilGitHub
