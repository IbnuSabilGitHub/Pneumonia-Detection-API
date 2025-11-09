# 🏥 Pneumonia Detection API

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [API Endpoints](#api-endpoints)
4. [Authentication & Security](#authentication--security)
5. [Request/Response Examples](#requestresponse-examples)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Model Information](#model-information)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## 🌟 Overview

The Pneumonia Detection API is an advanced AI-powered medical imaging analysis service that detects pneumonia from chest X-ray images using state-of-the-art deep learning models.

### ⚠️ Important Medical Disclaimer

**This API is for educational and research purposes only. The predictions should NEVER be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.**

### 🚀 Key Features

- **AI-Powered Detection**: Advanced deep learning models for pneumonia detection
- **Multiple Model Support**: Standard CNN and EfficientNet-B0 architectures
- **Confidence Scoring**: Detailed probability distributions and confidence levels
- **Medical Recommendations**: Contextual guidance based on AI predictions
- **Enterprise Security**: Multi-layer security with advanced rate limiting
- **Comprehensive Validation**: File size, type, and content validation
- **Real-time Monitoring**: Request logging and performance tracking

## 🚀 Quick Start

### Base URL
```
http://localhost:8000
```

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Basic Prediction
```bash
curl -X POST "http://localhost:8000/pneumonia/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"
```

## 📡 API Endpoints

### 🏥 Health & Monitoring

#### `GET /` or `GET /health`
**Service Health Check**

Returns comprehensive health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime": 3600.5
}
```

**Status Values:**
- `healthy`: All systems operational
- `partial`: Service running with limitations
- `unhealthy`: Critical issues detected

### 🔬 Pneumonia Detection

#### `POST /pneumonia/predict`
**AI-Powered Pneumonia Detection**

Analyzes chest X-ray images to detect pneumonia with confidence scoring.

**Parameters:**
- `file` (required): Chest X-ray image (JPG, JPEG, PNG, max 10MB)
- `model` (optional): AI model to use (`standard` or `efficientnet_b0`)

**Request Example:**
```bash
curl -X POST "http://localhost:8000/pneumonia/predict?model=efficientnet_b0" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"
```

**Response:**
```json
{
  "prediction": "NORMAL",
  "confidence": 0.92,
  "probabilities": {
    "NORMAL": 0.92,
    "PNEUMONIA": 0.08
  },
  "medical_recommendation": "✅ Normal results - maintain regular health checkups",
  "model_version": "v1.0",
  "model_type": "efficientnet_b0",
  "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
}
```

#### `GET /pneumonia/model/info`
**Model Information**

Returns detailed information about the loaded AI model.

**Parameters:**
- `model` (optional): Model to query (`standard` or `efficientnet_b0`)

**Response:**
```json
{
  "model_type": "efficientnet_b0",
  "model_version": "v1.0",
  "architecture": "EfficientNet-B0",
  "input_shape": [224, 224, 3],
  "classes": ["NORMAL", "PNEUMONIA"],
  "training_accuracy": 0.95,
  "validation_accuracy": 0.92,
  "inference_time_ms": 300,
  "model_size_mb": 20.5
}
```

### 🛡️ Security & Monitoring (Admin Only)

#### `GET /status` 🔒
**Security System Status**

**⚠️ AUTHENTICATION REQUIRED**: This endpoint requires admin API key authentication.

Returns real-time status of the security protection system.

**Request Headers:**
```http
X-Admin-API-Key: your-admin-api-key-here
```

**Request Example:**
```bash
curl -X GET "http://localhost:8000/status" \
     -H "X-Admin-API-Key: your-admin-api-key"
```

**Response:**
```json
{
  "service": "Pneumonia Detection API",
  "security_status": "active",
  "timestamp": "2025-11-09T10:30:00.000Z",
  "advanced_protection": {
    "global_attack_score": 0.15,
    "requests_per_minute": 23,
    "recent_unique_ips": 8,
    "blocked_fingerprints": 2,
    "storage_type": "memory"
  },
  "protection_features": [
    "Multi-layer Rate Limiting",
    "IP Switching Attack Detection",
    "Request Fingerprinting",
    "Behavioral Analysis",
    "Global Attack Scoring",
    "Duplicate File Detection"
  ]
}
```

**Error Responses:**
```json
// 401 Unauthorized - Missing API key
{
  "error": "Missing API key",
  "message": "Admin endpoints require X-Admin-API-Key header",
  "required_header": "X-Admin-API-Key"
}

// 403 Forbidden - Invalid API key
{
  "error": "Invalid API key",
  "message": "The provided API key is not valid"
}
```

#### `GET /stats` 🔒
**Detailed Security Statistics**

**⚠️ AUTHENTICATION REQUIRED**: This endpoint requires admin API key authentication.

Returns comprehensive security analytics and threat assessment.

**Request Headers:**
```http
X-Admin-API-Key: your-admin-api-key-here
```

**Request Example:**
```bash
curl -X GET "http://localhost:8000/stats" \
     -H "X-Admin-API-Key: your-admin-api-key"
```

**Response:**
```json
{
  "security_metrics": {
    "global_attack_score": 0.25,
    "requests_per_minute": 45,
    "recent_unique_ips": 12,
    "blocked_fingerprints": 3,
    "storage_type": "memory"
  },
  "timestamp": "2025-11-09T10:30:00.000Z",
  "interpretation": {
    "attack_score": {
      "value": 0.25,
      "level": "LOW",
      "description": "Global attack probability score (0.0-1.0)"
    },
    "request_rate": {
      "value": 45,
      "description": "Total requests in the last minute"
    }
  }
}
```

**Why are these endpoints protected?**

These endpoints are protected to prevent:
1. **Information Disclosure**: Attackers viewing real-time threat detection metrics
2. **Attack Intelligence Gathering**: Adversaries monitoring detection effectiveness
3. **System Profiling**: Exposure of internal performance and architecture
4. **Security Best Practice**: Aligns with industry standards (GitHub, AWS, Stripe)

**Who should have access?**
- ✅ **Admin/DevOps Teams**: Monitoring, debugging, incident response
- ✅ **Security Teams**: Threat analysis and protection tuning
- ❌ **Public Users**: No legitimate need for internal security metrics
- ❌ **API Consumers**: Should only see their own quotas, not global metrics

**Setup Instructions:**
```bash
# 1. Generate secure API key
openssl rand -hex 32

# 2. Set environment variable
export ADMIN_API_KEY="your-generated-key"

# 3. Restart service
docker-compose restart pneumonia-api

# 4. Test authentication
curl -H "X-Admin-API-Key: your-key" http://localhost:8000/status
```

For complete admin security documentation, see: [ADMIN_ENDPOINTS_SECURITY.md](ADMIN_ENDPOINTS_SECURITY.md)

## 🔐 Authentication & Security

### Admin Endpoint Authentication
- **Protected Endpoints**: `/status`, `/stats`
- **Method**: API Key in header (`X-Admin-API-Key`)
- **Required**: Yes (unless `ENABLE_PUBLIC_STATS=true` / `ENABLE_PUBLIC_STATUS=true`)
- **Key Generation**: `openssl rand -hex 32`

### Rate Limiting
- **Prediction Endpoints**: 5 requests per minute per IP
- **Other Endpoints**: No rate limits
- **Rate Limit Headers**: Included in responses

### File Validation
- **Supported Formats**: JPG, JPEG, PNG
- **Maximum Size**: 10MB
- **Content Validation**: AI-powered image validation
- **Duplicate Detection**: Prevents repeated uploads

### Security Features
- Multi-layer rate limiting
- IP switching attack detection
- Request fingerprinting
- Behavioral analysis
- Global attack scoring
- Comprehensive request logging

## 📝 Request/Response Examples

### Successful Prediction (Normal)
```bash
curl -X POST "http://localhost:8000/pneumonia/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@normal_chest_xray.jpg"
```

**Response (200 OK):**
```json
{
  "prediction": "NORMAL",
  "confidence": 0.94,
  "probabilities": {
    "NORMAL": 0.94,
    "PNEUMONIA": 0.06
  },
  "medical_recommendation": "✅ Normal results - maintain regular health checkups",
  "model_version": "v1.0",
  "model_type": "standard",
  "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
}
```

### Pneumonia Detected
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.85,
  "probabilities": {
    "NORMAL": 0.15,
    "PNEUMONIA": 0.85
  },
  "medical_recommendation": "⚠️ Possible pneumonia detected - seek immediate medical evaluation",
  "model_version": "v1.0",
  "model_type": "efficientnet_b0",
  "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
}
```

### Uncertain Result
```json
{
  "prediction": "NORMAL",
  "confidence": 0.55,
  "probabilities": {
    "NORMAL": 0.55,
    "PNEUMONIA": 0.45
  },
  "medical_recommendation": "❓ Uncertain results - professional medical evaluation recommended",
  "model_version": "v1.0",
  "model_type": "standard",
  "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
}
```

## ❌ Error Handling

### Common Error Responses

#### Invalid File Format (400)
```json
{
  "detail": "Unsupported file type. Allowed: jpg, jpeg, png",
  "error_code": "INVALID_FILE_FORMAT"
}
```

#### File Too Large (413)
```json
{
  "detail": "File size exceeds limit of 10.0 MB",
  "error_code": "FILE_TOO_LARGE"
}
```

#### Rate Limit Exceeded (429)
```json
{
  "detail": "Rate limit exceeded: 5 per minute",
  "error_code": "RATE_LIMIT_EXCEEDED"
}
```

#### Duplicate File (409)
```json
{
  "detail": "Duplicate file detected. Please wait before uploading the same image again.",
  "error_code": "DUPLICATE_FILE"
}
```

#### Service Unavailable (503)
```json
{
  "detail": "Prediction service is not available",
  "error_code": "SERVICE_UNAVAILABLE"
}
```

#### Invalid Image Content (400)
```json
{
  "detail": "Image does not appear to be a valid chest X-ray",
  "error_code": "INVALID_IMAGE_CONTENT"
}
```

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (validation errors)
- `404`: Not Found
- `409`: Conflict (duplicate files)
- `413`: Payload Too Large
- `429`: Too Many Requests
- `500`: Internal Server Error
- `503`: Service Unavailable

## ⏱️ Rate Limiting

### Limits
- **Prediction Endpoints**: 5 requests per minute per IP address
- **Security Endpoints**: No rate limits
- **Health Endpoints**: No rate limits

### Rate Limit Headers
```
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 3
X-RateLimit-Reset: 1625097600
```

### Best Practices
- Implement exponential backoff for rate limit errors
- Monitor rate limit headers in responses
- Cache results when appropriate
- Use appropriate request intervals

## 🤖 Model Information

### Available Models

#### Standard Model
- **Architecture**: Baseline CNN
- **Inference Time**: ~200ms
- **Accuracy**: ~91%
- **Use Case**: High-volume processing
- **Model Size**: ~15MB

#### EfficientNet-B0
- **Architecture**: Advanced transfer learning
- **Inference Time**: ~300ms
- **Accuracy**: ~95%
- **Use Case**: Critical analysis requiring higher accuracy
- **Model Size**: ~20MB

### Model Selection Guidelines
- Use **Standard** for: High-volume processing, quick screening
- Use **EfficientNet-B0** for: Critical analysis, higher accuracy requirements

## 💡 Best Practices

### Image Requirements
- **Format**: JPG/JPEG preferred for medical images
- **Quality**: High resolution, clear images
- **Orientation**: Frontal chest X-ray view
- **Size**: Under 10MB, ideally 2-5MB
- **Content**: Clean chest X-ray without artifacts

### API Usage
- Implement proper error handling for all endpoints
- Use appropriate models based on use case requirements
- Monitor rate limits and implement backoff strategies
- Validate responses and handle edge cases
- Log API interactions for debugging

### Security
- Never store sensitive medical data
- Implement HTTPS in production
- Monitor for unusual usage patterns
- Validate all inputs on client side
- Use secure file upload practices

### Performance
- Resize images appropriately before upload
- Use efficient file formats (JPEG over PNG for photos)
- Implement caching for repeated requests
- Monitor response times and optimize accordingly

## 🔧 Troubleshooting

### Common Issues

#### Model Not Loading
**Symptoms**: 503 errors, "model not loaded" in health check
**Solutions**: 
- Check model files exist in `/models` directory
- Verify ONNX Runtime installation
- Check memory availability
- Review application logs

#### High Response Times
**Symptoms**: Slow API responses
**Solutions**:
- Reduce image file sizes
- Use Standard model for faster inference
- Check system resources (CPU/Memory)
- Monitor concurrent requests

#### Rate Limit Issues
**Symptoms**: 429 errors
**Solutions**:
- Implement request throttling
- Use exponential backoff
- Monitor request frequency
- Distribute load across time

#### File Upload Failures
**Symptoms**: 400 errors on file upload
**Solutions**:
- Verify file format (JPG, JPEG, PNG only)
- Check file size (max 10MB)
- Ensure valid image content
- Validate image integrity

### Debugging Steps

1. **Check Health Endpoint**: Verify service status
2. **Review Logs**: Check application logs for errors
3. **Test with Known Good Images**: Use validated test images
4. **Monitor Security Status**: Check for blocks or attacks
5. **Verify Model Info**: Ensure correct model loading

### Contact & Support

For technical issues or questions:
- Check the GitHub repository: [Pneumonia-Detection-API](https://github.com/IbnuSabilGitHub/Pneumonia-Detection-API)
- Review documentation at `/docs` and `/redoc`
- Check system logs for detailed error information

---

**Built with FastAPI** | **Powered by ONNX** | **Secured by Design**
