# Pneumonia Detection API - Refactored Architecture

## Description
FastAPI application for pneumonia detection from chest X-ray images using machine learning. This API features a clean modular architecture, comprehensive security measures, and educational tools for learning medical AI development.

## 🏗️ Architecture Overview

This API has been completely refactored with a modern, scalable architecture:

- **Dependency Injection**: Testable and maintainable service components  
- **Security First**: Built-in rate limiting, validation, and security middleware
- **Configuration Management**: Environment-based settings with type safety
- **Comprehensive Logging**: Structured logging throughout the application
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and navigate to the project
cd Pneumonia-Detection-API

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
# Create .env file for custom configuration
cp .env.example .env
# Edit .env file to override default settings
```

### 3. Run the Application

```bash
# Method 1: Using FastAPI CLI (Recommended for development)
fastapi dev main.py

# Method 2: Using uvicorn directly with the refactored app
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

```

## Production Deployment

For production, use:

```bash
# Single worker
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Multiple workers (better performance)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

```

The API will be available at:
- **Main API**: `http://127.0.0.1:8000` or `http://localhost:8000`
- **Interactive Docs**: `http://127.0.0.1:8000/docs`
- **Alternative Docs**: `http://127.0.0.1:8000/redoc`

## 🏛️ Architecture Features

### ✅ **Clean Architecture**
- **Layered Design**: API → Services → Utils separation
- **Dependency Injection**: Testable and maintainable components  
- **Error Handling**: Structured exceptions and consistent error responses
- **Configuration**: Environment-based settings with validation

### ✅ **Security & Monitoring**
- **Rate Limiting**: 5 requests/minute per IP with temporary blocking
- **Input Validation**: File size, type, and medical image content checks
- **Duplicate Detection**: SHA-256 hash-based duplicate prevention
- **Request Logging**: Comprehensive logging with IP tracking
- **Security Middleware**: Custom middleware for security and monitoring

### ✅ **Developer Experience**
- **API Documentation**: Auto-generated OpenAPI/Swagger at `/docs`
- **Health Checks**: Service health monitoring at `/`
- **Security Status**: Rate limiting and security info at `/security/status`
- **Modular Code**: Clean, readable, and maintainable codebase

## 🔗 API Endpoints

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `GET` | `/` | Health check and service status | None |
| `POST` | `/pneumonia/predict` | Pneumonia detection from X-ray | 5/min |
| `GET` | `/pneumonia/model/info` | Model information and stats | None |
| `GET` | `/security/status` | Security and rate limiting status | 10/min |
| `GET` | `/docs` | Interactive API documentation | None |
| `GET` | `/redoc` | Alternative API documentation | None |

## 🧪 Testing the API

### Using cURL
```bash
# Test health check
curl http://localhost:8000/

# Upload chest X-ray for prediction
curl -X POST "http://localhost:8000/pneumonia/predict" \
  -F "file=@sample_xray.jpg"

# Check security status
curl http://localhost:8000/security/status

# Get model information
curl http://localhost:8000/pneumonia/model/info
```

### Using Python
```python
import requests

# Test prediction
with open('chest_xray.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/pneumonia/predict',
        files={'file': f}
    )
    
print(response.json())
```

### Using Postman
1. Set method to `POST`
2. URL: `http://localhost:8000/pneumonia/predict`
3. Body: form-data
4. Key: `file` (type: File)
5. Value: Select your chest X-ray image

## 📊 Response Format

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

## 🛡️ Security Features (Educational)

| Feature | Purpose | Educational Value |
|---------|---------|-------------------|
| Rate Limiting | Prevent abuse | Learn about API throttling |
| File Validation | Security & performance | Input validation best practices |
| Duplicate Detection | Efficiency | Caching and hash-based deduplication |
| Request Logging | Monitoring | API observability and debugging |
| IP Tracking | Audit trail | User activity monitoring |

## 🔍 Monitoring & Debugging

### Check Security Status
```bash
curl http://localhost:8000/security/status
```

Response:
```json
{
  "client_ip": "127.0.0.1",
  "requests_in_last_minute": 2,
  "rate_limit": "5 requests per minute",
  "is_blocked": false,
  "cache_entries": 3,
  "security_features": [
    "Rate Limiting (5/min per IP)",
    "File Size Validation (10MB max)",
    "File Type Validation (JPG, JPEG, PNG)",
    "Image Content Validation",
    "Duplicate Detection (5min cache)",
    "Request Logging with IP tracking",
    "Educational/Learning Mode (No Auth Required)"
  ]
}
```

## ⚠️ Important Educational Notes

### 🎯 **This is for Learning Only**
- Model is not clinically validated
- Results should not be used for medical decisions
- Always include appropriate disclaimers
- Focus on technical learning aspects

### 🔧 **Development Best Practices Demonstrated**
- Input validation and sanitization
- Rate limiting and abuse prevention
- Error handling and logging
- API documentation and testing
- Security monitoring
- Clean code structure

### 📈 **Next Steps for Learning**
1. **Add Authentication** - Learn OAuth2, JWT
2. **Database Integration** - Store results, user data
3. **Async Processing** - Background tasks, queues
4. **Testing** - Unit tests, integration tests
5. **Deployment** - Docker, Railway, monitoring
6. **Advanced Security** - HTTPS, CORS, headers

## 🐛 Common Issues & Solutions

### Rate Limit Hit
- Wait 1 minute and try again
- Check `/security/status` for current status
- Each IP gets 5 requests per minute

### File Upload Errors
- Ensure file is JPG, JPEG, or PNG
- Maximum file size: 10MB
- File should be a chest X-ray image

### Connection Issues
- Ensure server is running on `localhost:8000`
- Check firewall settings
- Verify virtual environment is activated

## 📝 Logs Location

Application logs show:
- Request details (IP, file info)
- Prediction results
- Security events (rate limits, blocks)
- Error information for debugging

Perfect for learning how to monitor and debug APIs!

## 🤝 Contributing to Learning

This educational API demonstrates:
- Modern FastAPI development
- Security best practices (simplified)
- ML model serving
- API documentation
- Error handling
- Monitoring and observability

Feel free to experiment, modify, and extend for your learning purposes! 🎉
