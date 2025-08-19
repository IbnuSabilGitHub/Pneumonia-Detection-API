# Pneumonia Detection API - Architecture Documentation

## Overview

The Pneumonia Detection API has been completely refactored to provide a scalable, maintainable, and clean architecture. This document outlines the new modular design and architectural decisions.

## Architecture Principles

- **Separation of Concerns**: Each module has a single, well-defined responsibility
- **Dependency Injection**: Services are injected rather than globally instantiated
- **Configuration Management**: Centralized settings with environment variable support
- **Error Handling**: Structured exception handling with custom error types
- **Logging**: Comprehensive logging throughout the application
- **Security**: Built-in rate limiting, validation, and security middleware
- **Testability**: Modular design enables easy unit testing

## Project Structure

```
pneumonia-detection-api/
├── app/                           # Main application package
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # FastAPI application factory
│   ├── core/                     # Core application components
│   │   ├── __init__.py
│   │   ├── settings.py           # Configuration and settings
│   │   └── logger.py             # Logging configuration
│   ├── api/                      # API route handlers
│   │   ├── __init__.py
│   │   ├── health.py             # Health check endpoints
│   │   └── prediction.py         # Prediction endpoints
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   └── prediction.py         # ML prediction service
│   ├── models/                   # Pydantic models
│   │   ├── __init__.py
│   │   └── schemas.py            # Request/response schemas
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── exceptions.py         # Custom exceptions
│   │   ├── security.py           # Security utilities
│   │   └── validation.py         # Input validation
│   └── middleware/               # Custom middleware
│       ├── __init__.py
│       └── security.py           # Security middleware
├── models/                       # ML model files
│   ├── pneumonia_model.onnx      # ONNX model
│   └── model_stats.json          # Model normalization stats
├── main.py                       # Entry point for the Pneumonia Detection API.
├── requirements.txt              # Python dependencies
├── requirements_dev.txt          # Python dependencies for development
├── Dockerfile                    # Container configuration
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

## Layer Architecture

### 1. **API Layer** (`app/api/`)
- **Responsibility**: Handle HTTP requests and responses
- **Components**:
  - `health.py`: Health check and monitoring endpoints
  - `prediction.py`: Pneumonia detection endpoints
- **Features**:
  - Request validation using Pydantic models
  - Rate limiting with SlowAPI
  - Comprehensive error handling
  - API documentation with OpenAPI/Swagger

### 2. **Service Layer** (`app/services/`)
- **Responsibility**: Business logic and ML operations
- **Components**:
  - `prediction.py`: Pneumonia prediction service
- **Features**:
  - Model loading and management
  - Image preprocessing
  - Inference execution
  - Result post-processing

### 3. **Core Layer** (`app/core/`)
- **Responsibility**: Application configuration and shared utilities
- **Components**:
  - `settings.py`: Centralized configuration management
  - `logger.py`: Logging setup and configuration
- **Features**:
  - Environment-based configuration
  - Type-safe settings with Pydantic
  - Structured logging

### 4. **Utils Layer** (`app/utils/`)
- **Responsibility**: Reusable utility functions
- **Components**:
  - `exceptions.py`: Custom exception classes
  - `security.py`: Security utilities (rate limiting, IP handling)
  - `validation.py`: Input validation functions
- **Features**:
  - File validation
  - Image content validation
  - Security utilities
  - Custom exceptions for better error handling

### 5. **Middleware Layer** (`app/middleware/`)
- **Responsibility**: Request/response processing
- **Components**:
  - `security.py`: Security middleware
- **Features**:
  - Rate limiting
  - Request logging
  - Global error handling

### 6. **Models Layer** (`app/models/`)
- **Responsibility**: Data models and schemas
- **Components**:
  - `schemas.py`: Pydantic models for API contracts
- **Features**:
  - Request/response validation
  - API documentation generation
  - Type safety

## Key Architectural Improvements

### 1. **Configuration Management**
- Centralized settings in `app/core/settings.py`
- Environment variable support with `.env` files
- Type-safe configuration using Pydantic
- Separate configurations for different environments

### 2. **Dependency Injection**
- Services are injected into routes using FastAPI's dependency system
- Enables easy testing and mocking
- Loose coupling between components

### 3. **Error Handling**
- Custom exception hierarchy for different error types
- Global exception middleware for consistent error responses
- Structured error responses with error codes

### 4. **Security Enhancements**
- In-memory rate limiting with IP blocking
- File validation and content checking
- Duplicate detection using file hashes
- Security middleware for request logging

### 5. **Logging and Monitoring**
- Structured logging with configurable levels
- Request/response logging
- Performance monitoring
- Error tracking with stack traces

### 6. **Scalability Features**
- Modular architecture for easy extension
- Service-oriented design
- Configurable components
- Support for different deployment environments

## Data Flow

```
Request → Middleware → API Routes → Services → Utils → Response
    ↓         ↓           ↓           ↓         ↓
 Security   Logging   Validation   ML Logic  Validation
 Checks              Pydantic     ONNX Model  File/Image
```

### Request Processing Flow:

1. **Middleware Processing**:
   - Security middleware checks rate limits
   - Logging middleware records request details
   - Error handling middleware catches exceptions

2. **API Route Processing**:
   - Route validates request using Pydantic models
   - Dependency injection provides required services
   - Business logic is delegated to services

3. **Service Processing**:
   - Prediction service handles ML operations
   - Image preprocessing and validation
   - Model inference and result processing

4. **Response Generation**:
   - Results are formatted using Pydantic response models
   - Errors are handled consistently
   - Logging captures response details

## Security Architecture

### 1. **Rate Limiting**
- In-memory rate limiter with configurable limits
- IP-based tracking with temporary blocking
- Separate limits for different endpoints

### 2. **Input Validation**
- File size and type validation
- Image content validation for medical relevance
- Duplicate detection using file hashes

### 3. **Security Headers**
- Trusted host middleware
- CORS configuration
- Request logging with IP tracking

### 4. **Error Handling**
- No sensitive information in error responses
- Structured error codes for debugging
- Comprehensive logging for security monitoring

## Performance Considerations

### 1. **Model Loading**
- Model loaded once at application startup
- In-memory model for fast inference
- Graceful error handling for model loading failures

### 2. **Caching**
- File hash caching for duplicate detection
- Configurable cache durations
- Memory-efficient cache cleanup

### 3. **Request Processing**
- Asynchronous request handling with FastAPI
- Efficient image processing with PIL/NumPy
- Minimal memory footprint for image operations

## Testing Strategy

### 1. **Unit Testing**
- Individual service testing
- Utility function testing
- Mock dependencies for isolated testing

### 2. **Integration Testing**
- API endpoint testing
- End-to-end workflow testing
- Database/model integration testing

### 3. **Load Testing**
- Rate limiting validation
- Performance under load
- Memory usage monitoring

## Deployment Considerations

### 1. **Environment Configuration**
- Environment-specific settings
- Container-ready configuration
- Health checks for monitoring

### 2. **Scaling**
- Stateless application design
- Horizontal scaling support
- Load balancer compatibility

### 3. **Monitoring**
- Health check endpoints
- Structured logging for log aggregation
- Performance metrics collection

## Future Enhancements

### 1. **Database Integration**
- User management and authentication
- Request history and analytics
- Model performance tracking

### 2. **Advanced Security**
- JWT-based authentication
- API key management
- Advanced rate limiting with Redis

### 3. **Model Management**
- Model versioning
- A/B testing support
- Model performance monitoring

### 4. **API Features**
- Batch processing
- Async processing for large files
- WebSocket support for real-time updates

## Configuration Reference

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APP_NAME` | Application name | "Pneumonia Detection API" | No |
| `APP_VERSION` | Application version | "2.0.0" | No |
| `DEBUG` | Debug mode | False | No |
| `HOST` | Server host | "0.0.0.0" | No |
| `PORT` | Server port | 8000 | No |
| `MODEL_PATH` | Path to ONNX model | "models/pneumonia_model.onnx" | No |
| `MODEL_STATS_PATH` | Path to model stats | "models/model_stats.json" | No |
| `RATE_LIMIT_REQUESTS` | Rate limit per window | 5 | No |
| `RATE_LIMIT_WINDOW` | Rate limit window (seconds) | 60 | No |
| `MAX_FILE_SIZE` | Maximum file size (bytes) | 10485760 | No |
| `LOG_LEVEL` | Logging level | "INFO" | No |

## Conclusion

The refactored architecture provides a solid foundation for a production-ready medical AI API with proper separation of concerns, security features, and scalability considerations. The modular design enables easy maintenance, testing, and future enhancements while maintaining clean code principles and following FastAPI best practices.
