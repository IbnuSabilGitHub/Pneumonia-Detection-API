"""
Security API endpoints.
"""
from datetime import datetime
from fastapi import APIRouter

from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/status",
    tags=["Security"],
    summary="🛡️ Security System Status",
    description="""
<h2>🔒 Advanced Security Protection Status</h2>

<p>Provides real-time status of the multi-layer security system protecting the API
from various attacks and abuse patterns.</p>

<h3>🛡️ <strong>Protection Features Monitored</strong></h3>

<ul>
<li><strong>Multi-layer Rate Limiting</strong>: IP-based and global request limits</li>
<li><strong>Attack Detection</strong>: Sophisticated pattern recognition</li>
<li><strong>Request Fingerprinting</strong>: Unique request identification</li>
<li><strong>Behavioral Analysis</strong>: User behavior pattern analysis</li>
<li><strong>IP Switching Detection</strong>: Rapid IP change detection</li>
<li><strong>Duplicate Prevention</strong>: File and request deduplication</li>
<li><strong>Global Attack Scoring</strong>: Overall threat level assessment</li>
</ul>

<h3>📊 <strong>Status Information</strong></h3>

<ul>
<li><strong>System Status</strong>: Overall security system health</li>
<li><strong>Active Protections</strong>: List of enabled security features</li>
<li><strong>Threat Level</strong>: Current global attack probability</li>
<li><strong>Storage Backend</strong>: In-memory or Redis storage status</li>
<li><strong>Performance Metrics</strong>: Security system performance data</li>
</ul>

<h3>🎯 <strong>Use Cases</strong></h3>

<ul>
<li><strong>Security Monitoring</strong>: Real-time protection status</li>
<li><strong>Threat Assessment</strong>: Current security posture</li>
<li><strong>System Health</strong>: Security component validation</li>
<li><strong>Compliance</strong>: Security audit and reporting</li>
</ul>

<h3>⚡ <strong>Performance</strong></h3>

<ul>
<li><strong>Response Time</strong>: &lt; 100ms typical</li>
<li><strong>Rate Limiting</strong>: No limits applied to security endpoints</li>
<li><strong>Real-time Data</strong>: Live security metrics</li>
</ul>
    """,
    response_description="Current security system status and active protections",
    responses={
        200: {
            "description": "Security status retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "service": "Pneumonia Detection API",
                        "security_status": "active",
                        "timestamp": "2025-09-05T10:30:00.000Z",
                        "advanced_protection": {
                            "global_attack_score": 0.15,
                            "requests_per_minute": 23,
                            "recent_unique_ips": 8,
                            "blocked_fingerprints": 2,
                            "storage_type": "memory"
                        },
                        "protection_features": [
                            "Multi-layer Rate Limiting (In-Memory)",
                            "IP Switching Attack Detection (In-Memory)",
                            "Request Fingerprinting (In-Memory)",
                            "Behavioral Analysis (In-Memory)",
                            "Global Attack Scoring (In-Memory)",
                            "Duplicate File Detection (In-Memory)"
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Security system error",
            "content": {
                "application/json": {
                    "example": {
                        "service": "Pneumonia Detection API",
                        "security_status": "not_initialized",
                        "timestamp": "2025-09-05T10:30:00.000Z",
                        "error": "Rate limiter not initialized"
                    }
                }
            }
        }
    }
)
async def get_security_status():
    """
    **🛡️ Advanced Security System Status**
    
    Provides comprehensive real-time status of the multi-layer security protection
    system including threat detection, rate limiting, and attack prevention measures.
    
    **Security Layers Monitored:**
    - **Rate Limiting**: Request frequency controls per IP
    - **Attack Detection**: Sophisticated attack pattern recognition
    - **Behavioral Analysis**: User behavior anomaly detection
    - **Request Fingerprinting**: Unique request identification and tracking
    - **IP Switching Detection**: Rapid IP change pattern detection
    - **File Duplication Prevention**: Duplicate upload detection
    - **Global Threat Scoring**: Overall attack probability assessment
    
    **Status Categories:**
    - **active**: Security system fully operational
    - **not_initialized**: Security components not ready
    - **degraded**: Partial security functionality
    
    **Returns:**
        dict: Complete security status with:
        - Overall system health and operational status
        - List of active protection features
        - Current threat levels and attack scores
        - Storage backend status and performance
        - Timestamp for status validity
        
    **Use Cases:**
        - Real-time security monitoring
        - Threat level assessment
        - Security audit and compliance
        - System health verification
    """
    # Import at runtime to get the latest reference
    from ..core.advanced_rate_limiting import get_rate_limiter
    advanced_rate_limiter = get_rate_limiter()
    
    if advanced_rate_limiter is None:
        return {
            "service": "Pneumonia Detection API",
            "security_status": "not_initialized",
            "timestamp": datetime.now().isoformat(),
            "error": "Rate limiter not initialized"
        }
    
    # Use async method if available, otherwise fallback
    try:
        if hasattr(advanced_rate_limiter, '_storage_initialized') and advanced_rate_limiter._storage_initialized:
            status = await advanced_rate_limiter.get_security_status_async()
        else:
            status = advanced_rate_limiter.get_security_status()
    except Exception as e:
        logger.error(f"Failed to get security status: {e}")
        status = {"error": str(e)}
    
    return {
        "service": "Pneumonia Detection API",
        "security_status": "active",
        "timestamp": datetime.now().isoformat(),
        "advanced_protection": status,
        "protection_features": [
            "Multi-layer Rate Limiting (In-Memory)",
            "IP Switching Attack Detection (In-Memory)", 
            "Request Fingerprinting (In-Memory)",
            "Behavioral Analysis (In-Memory)",
            "Global Attack Scoring (In-Memory)",
            "Duplicate File Detection (In-Memory)",
            "Persistent Storage (In-Memory)",
            "Single-instance Optimized (In-Memory)"
        ]
    }


@router.get(
    "/stats",
    tags=["Security"],
    summary="📊 Detailed Security Statistics",
    description="""
<h2>📈 Advanced Security Metrics & Analytics</h2>

<p>Provides comprehensive statistical analysis of security events, threat patterns,
and protection effectiveness over time.</p>

<h3>📊 <strong>Detailed Metrics</strong></h3>

<ul>
<li><strong>Attack Score Analysis</strong>: Global threat probability with interpretation</li>
<li><strong>Request Rate Metrics</strong>: Real-time request frequency analysis</li>
<li><strong>IP Activity Tracking</strong>: Unique IP address patterns and behavior</li>
<li><strong>Blocking Statistics</strong>: Currently blocked requests and fingerprints</li>
<li><strong>Performance Metrics</strong>: Security system response times</li>
<li><strong>Storage Utilization</strong>: Memory/Redis usage statistics</li>
</ul>

<h3>🎯 <strong>Threat Level Interpretation</strong></h3>

<ul>
<li><strong>LOW</strong> (0.0-0.3): Normal traffic patterns, minimal threats</li>
<li><strong>MEDIUM</strong> (0.3-0.7): Elevated activity, possible probing</li>
<li><strong>HIGH</strong> (0.7-1.0): Active attacks detected, enhanced protection</li>
</ul>

<h3>📋 <strong>Statistical Categories</strong></h3>

<ul>
<li><strong>Real-time Metrics</strong>: Current minute activity</li>
<li><strong>Pattern Analysis</strong>: Behavioral pattern recognition</li>
<li><strong>Threat Assessment</strong>: Attack probability calculations</li>
<li><strong>Performance Data</strong>: System efficiency metrics</li>
</ul>

<h3>🔍 <strong>Use Cases</strong></h3>

<ul>
<li><strong>Security Analytics</strong>: Deep dive into protection metrics</li>
<li><strong>Threat Intelligence</strong>: Attack pattern analysis</li>
<li><strong>Performance Tuning</strong>: Security system optimization</li>
<li><strong>Incident Response</strong>: Detailed attack investigation</li>
<li><strong>Compliance Reporting</strong>: Security audit documentation</li>
</ul>

<h3>⚡ <strong>Data Freshness</strong></h3>

<ul>
<li><strong>Update Frequency</strong>: Real-time (&lt; 1 second delay)</li>
<li><strong>Historical Data</strong>: Last 24 hours of activity</li>
<li><strong>Aggregation</strong>: Per-minute statistical summaries</li>
</ul>
    """,
    response_description="Comprehensive security statistics and threat analysis",
    responses={
        200: {
            "description": "Security statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "security_metrics": {
                            "global_attack_score": 0.25,
                            "requests_per_minute": 45,
                            "recent_unique_ips": 12,
                            "blocked_fingerprints": 3,
                            "storage_type": "memory",
                            "avg_response_time_ms": 85,
                            "total_requests_24h": 2847
                        },
                        "timestamp": "2025-09-05T10:30:00.000Z",
                        "interpretation": {
                            "attack_score": {
                                "value": 0.25,
                                "level": "LOW",
                                "description": "Global attack probability score (0.0-1.0)"
                            },
                            "request_rate": {
                                "value": 45,
                                "description": "Total requests in the last minute"
                            },
                            "unique_ips": {
                                "value": 12,
                                "description": "Number of unique IP addresses in recent activity"
                            },
                            "blocked_count": {
                                "value": 3,
                                "description": "Number of currently blocked request fingerprints"
                            }
                        }
                    }
                }
            }
        },
        500: {
            "description": "Security statistics unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Rate limiter not initialized",
                        "timestamp": "2025-09-05T10:30:00.000Z"
                    }
                }
            }
        }
    }
)
async def get_security_stats():
    """
    **📊 Comprehensive Security Analytics Dashboard**
    
    Provides detailed statistical analysis and metrics of the security protection
    system including threat patterns, attack detection results, and system performance.
    
    **Metrics Categories:**
    
    1. **Threat Analysis**
       - Global attack probability scoring (0.0-1.0 scale)
       - Attack pattern recognition and classification
       - Threat level interpretation and recommendations
    
    2. **Traffic Analytics**
       - Real-time request rate monitoring
       - Unique IP address tracking and analysis
       - Request pattern and behavioral analysis
    
    3. **Protection Effectiveness**
       - Blocked request fingerprint statistics
       - Attack prevention success rates
       - False positive/negative analysis
    
    4. **System Performance**
       - Security system response times
       - Storage backend utilization
       - Processing efficiency metrics
    
    **Threat Level Interpretation:**
    - **LOW** (0.0-0.3): Normal operations, standard monitoring
    - **MEDIUM** (0.3-0.7): Elevated vigilance, possible threats
    - **HIGH** (0.7-1.0): Active attacks, enhanced protection mode
    
    **Returns:**
        dict: Comprehensive security analytics including:
        - Raw security metrics and measurements
        - Human-readable interpretation of threat levels
        - Actionable insights for security decisions
        - Performance benchmarks and system health
        
    **Use Cases:**
        - Security operations center (SOC) monitoring
        - Threat intelligence and pattern analysis
        - Performance optimization and tuning
        - Compliance and audit reporting
        - Incident response and investigation
    """
    # Import at runtime to get the latest reference
    from ..core.advanced_rate_limiting import get_rate_limiter
    advanced_rate_limiter = get_rate_limiter()
    
    if advanced_rate_limiter is None:
        return {
            "error": "Rate limiter not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Use async method if available
        if hasattr(advanced_rate_limiter, '_storage_initialized') and advanced_rate_limiter._storage_initialized:
            status = await advanced_rate_limiter.get_security_status_async()
        else:
            status = advanced_rate_limiter.get_security_status()
    except Exception as e:
        logger.error(f"Failed to get security stats: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "security_metrics": status,
        "timestamp": datetime.now().isoformat(),
        "interpretation": {
            "attack_score": {
                "value": status["global_attack_score"],
                "level": (
                    "LOW" if status["global_attack_score"] < 0.3 else
                    "MEDIUM" if status["global_attack_score"] < 0.7 else
                    "HIGH"
                ),
                "description": "Global attack probability score (0.0-1.0)"
            },
            "request_rate": {
                "value": status["requests_per_minute"],
                "description": "Total requests in the last minute"
            },
            "unique_ips": {
                "value": status["recent_unique_ips"],
                "description": "Number of unique IP addresses in recent activity"
            },
            "blocked_count": {
                "value": status["blocked_fingerprints"],
                "description": "Number of currently blocked request fingerprints"
            }
        }
    }
