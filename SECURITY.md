# Pneumonia Detection API - Security Documentation

## Security Features for Learning & Experimentation

This API is configured for **educational and experimental purposes** with important security features. The following are the main security features implemented:

### 1. Rate Limiting
- **Endpoint Protection**: `/pneumonia/predict` is protected with rate limiting
- **Limit**: 5 requests per minute per IP address
- **Implementation**: Dual protection with SlowAPI and custom in-memory rate limiter
- **Blocking**: IPs exceeding limits are temporarily blocked for 5 minutes
- **Purpose**: Prevent abuse while allowing learning experimentation

### 2. File Validation
- **Size Limit**: Maximum 10MB per file
- **File Types**: Only JPG, JPEG, PNG allowed
- **Content Validation**: Basic X-ray image characteristics validation
- **Integrity Check**: Image file validation before processing
- **Purpose**: Ensure only appropriate medical images are processed

### 3. Duplicate Detection
- **Hash-based**: SHA-256 file content hashing
- **Cache Duration**: 5 minutes
- **Purpose**: Prevent spam uploads of identical images for learning efficiency

### 4. Request Monitoring
- **IP Tracking**: All requests logged with client IP
- **Request Logging**: Detailed logging of predictions and security events
- **Security Status**: `/security/status` endpoint for monitoring
- **Purpose**: Learning about API monitoring and debugging

### 5. Educational Safety Features
- **Medical Disclaimers**: Clear warnings about experimental nature
- **Learning-focused**: Security without complexity barriers
- **Development-friendly**: Easy testing and experimentation

## Configuration

### Environment Variables

```bash
# Server Configuration
PORT=8000

# CORS for learning (allow local development)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Rate Limiting (reasonable for learning)
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_WINDOW=60

# Optional Redis for advanced learning
REDIS_URL=redis://localhost:6379
```

## API Usage (No Authentication Required)

### Basic Request
```bash
curl -X POST "http://localhost:8000/pneumonia/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

### Security Status Check
```bash
curl -X GET "http://localhost:8000/security/status"
```

### Example Response
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.89,
  "probabilities": {
    "NORMAL": 0.11,
    "PNEUMONIA": 0.89
  },
  "medical_recommendation": "⚠️ Medical consultation strongly recommended",
  "model_version": "v1.0",
  "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
}
```

## Security Response Headers

The API includes security headers for monitoring:

```json
{
  "client_ip": "192.168.1.100",
  "requests_in_last_minute": 3,
  "rate_limit": "5 requests per minute",
  "is_blocked": false,
  "cache_entries": 15,
  "security_features": [
    "Rate Limiting (5/min per IP)",
    "File Size Validation (10MB max)",
    "File Type Validation (JPG, JPEG, PNG)",
    "Image Content Validation",
    "Duplicate Detection (5min cache)",
    "Optional API Key Authentication",
    "Request Logging with IP tracking"
  ]
}
```

## Error Responses

### Rate Limit Exceeded (429)
```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

### Duplicate File (409)
```json
{
  "detail": "Duplicate file detected. Please wait before uploading the same image again."
}
```

### Invalid API Key (401)
```json
{
  "detail": "Invalid or missing API key"
}
```

### Invalid Image Content (400)
```json
{
  "detail": "Image does not appear to be a valid chest X-ray"
}
```

## Production Deployment

### Railway Environment Variables
Set these in your Railway dashboard:

1. `API_KEY` - Your secret API key (optional but recommended)
2. `ALLOWED_ORIGINS` - Your frontend domain
3. `RATE_LIMIT_REQUESTS` - Requests per minute (default: 5)
4. `RATE_LIMIT_WINDOW` - Time window in seconds (default: 60)

### Redis for Production Rate Limiting
For high-traffic production environments, consider using Redis for distributed rate limiting:

1. Add Redis service to Railway
2. Set `REDIS_URL` environment variable
3. The app will automatically use Redis if available

## Security Best Practices

1. **Always use HTTPS** in production
2. **Set strong API keys** (32+ characters)
3. **Monitor rate limiting logs** for abuse patterns
4. **Regular security updates** of dependencies
5. **Implement proper logging** and monitoring
6. **Use reverse proxy** (Nginx) for additional security layers

## Monitoring and Alerts

Monitor these metrics:
- Rate limit violations per IP
- Failed authentication attempts
- Unusual file upload patterns
- Server response times
- Error rates by endpoint

Consider setting up alerts for:
- High rate limit violations
- Multiple authentication failures
- Unusual traffic spikes
- Server errors
