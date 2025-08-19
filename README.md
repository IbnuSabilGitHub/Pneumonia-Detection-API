# Deteksi Pneumonia Menggunakan Gambar Rontgen Dada - API


## Description
This API is configured to detect pneumonia from chest X-ray images using a machine learning model. It is designed for educational purposes, allowing users to experiment with image uploads and predictions.
## 🎓 Overview

This is a **learning-focused** FastAPI application for pneumonia detection from chest X-ray images. It's designed for experimentation, learning, and educational purposes.

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work for learning)
```

### 3. Run the Application
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📚 Educational Features

### ✅ **Simple & Accessible**
- No complex authentication barriers
- Easy to test and experiment with
- Clear educational disclaimers
- Comprehensive logging for learning

### ✅ **Essential Security (Learning-focused)**
- **Rate Limiting**: 5 requests/minute per IP
- **File Validation**: Size, type, and content checks
- **Duplicate Detection**: Prevents spam uploads
- **Request Monitoring**: Learn about API monitoring

### ✅ **Learning-friendly Endpoints**
- `GET /` - Health check
- `POST /pneumonia/predict` - Main prediction endpoint
- `GET /docs` - Interactive API documentation
- `GET /security/status` - Security monitoring

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
