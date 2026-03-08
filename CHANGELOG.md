# Changelog

## [3.7.0] - 2026-03-08 - Complete Redis Removal

### Breaking Changes
- **REMOVED**: Complete Redis support and all related code
  - Deleted `redis_storage.py` module entirely
  - Removed Redis from storage factory and backend options
  - Cleaned up all Redis imports and references
  - Removed Redis from dependencies (requirements.txt)
  - Eliminated Redis configuration from settings and .env.example

### Architecture Simplification  
- **Storage Backend**: Now only supports `memory` and `database` (future)
  - Simplified from 3 options (memory, redis, database) to 2
  - Default remains `memory` for optimal single-instance deployment
  - Removed optional Redis scaling path
- **Rate Limiting**: Default storage changed from Redis to memory in rate_limiting/api.py
- **Configuration**: Removed all Redis-related environment variables:
  - `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_DB`
  - `REDIS_URL`, `REDIS_MAX_CONNECTIONS`, `REDIS_CLUSTER_MODE`
  - `REDIS_CLUSTER_NODES`, `REDIS_KEY_PREFIX`

---

## [3.6.2] - 2026-03-08 - fix: Add missing @staticmethod, fix rate limiting imports, improve fallback response
- Fix Missing @staticmethod in health_metadata.py
- Fix Import Errors (Root Cause)
- Improve Fallback Endpoint in main.py

## [3.6.1] - 2026-03-07 -  Improve code quality and organization

- Add Redis feature detection at module level in storage_factory.py
- Standardize imports to relative paths in rate_limiting module
- Reorganize and complete .env.example with all config variables
- Review auth module structure (no changes needed - already well organized)
- Add comprehensive refactoring documentation

Changes:
- storage_factory.py: Better Redis availability handling with clear error messages
- rate_limiting/*.py: Convert absolute imports to relative imports
- .env.example: Add missing variables, organize with section headers
- Add REFACTOR_COMPLETED.md for detailed change documentation


## [3.6.0] - 2026-03-04 - Supabase JWT Authentication Integration

### ✨ Major Features
- **NEW**: Comprehensive Supabase JWT authentication system
  - Automatic JWKS public key fetching for production tokens
  - Bearer token validation with configurable token expiry checks
  - User extraction with role-based access control

- **PROTECTED**: `/pneumonia/predict` endpoint now requires JWT authentication
  - Prevents unauthorized access to core prediction service
  - Includes user audit logging with Supabase user ID
  - Maintains request fingerprints with authenticated user context

- **ENHANCED**: `/security/status` and `/security/stats` endpoints
  - Support dual authentication: JWT (admin role) **OR** legacy API key
  - Backward compatible with existing API key integrations
  - Improved admin role detection (checks top-level role, app_metadata, user_metadata)

### ⚠️ Breaking Changes
- `/pneumonia/predict` now requires `Authorization: Bearer <access_token>` header
- Clients must obtain JWT from Supabase before calling predict endpoint
- Existing API integrations without JWT support will receive 401/503 errors

---

## [3.5.1] - 2025-11-09 - Admin Endpoints Security Enhancement

### 🔒 Security Improvements
- **BREAKING**: Admin endpoints `/stats` and `/status` now require authentication
  - Previously public endpoints are now protected with API key authentication
  - Prevents information disclosure and attack intelligence gathering
  - Implements "defense in depth" security principle

### ✨ Added
- **NEW**: Admin API key authentication system (`app/utils/auth.py`)
  - Header-based authentication: `X-Admin-API-Key`
  - Constant-time comparison to prevent timing attacks
  - Graceful degradation when admin key not configured (503 Service Unavailable)
- **NEW**: Configuration options in `settings.py`:
  - `ADMIN_API_KEY`: Required API key for admin endpoints (generated via `openssl rand -hex 32`)
  - `ENABLE_PUBLIC_STATS`: Optional flag to allow public access (default: `false`, not recommended)
  - `ENABLE_PUBLIC_STATUS`: Optional flag to allow public access (default: `false`, not recommended)
- **NEW**: Comprehensive admin security documentation (`doc/ADMIN_ENDPOINTS_SECURITY.md`)
  - API key generation guide
  - Usage examples with curl
  - Security best practices
  - Migration guide from public to private endpoints
  - Troubleshooting common issues

### 🔧 Changed
- **MODIFIED**: `/stats` endpoint now requires `X-Admin-API-Key` header (unless `ENABLE_PUBLIC_STATS=true`)
- **MODIFIED**: `/status` endpoint now requires `X-Admin-API-Key` header (unless `ENABLE_PUBLIC_STATUS=true`)
- **UPDATED**: Docker Compose configuration with `ADMIN_API_KEY` environment variable
- **UPDATED**: `.env.example` with admin security configuration examples

### 🛡️ Security Rationale
**Why protect these endpoints?**
1. **Information Disclosure Prevention**: Attackers cannot view real-time threat metrics
2. **Attack Intelligence Mitigation**: Prevents adversaries from monitoring detection effectiveness
3. **System Profiling Protection**: Hides internal system performance and architecture details
4. **Industry Best Practice**: Aligns with GitHub, AWS, and Stripe security models

### 📊 Impact Analysis
**Who needs access to `/stats` and `/status`?**
- ✅ **Admin/DevOps Teams**: For monitoring, debugging, and incident response
- ✅ **Security Teams**: For threat analysis and security tuning
- ❌ **Public Users**: No legitimate need for internal security metrics
- ❌ **API Consumers**: Should only see their own quota (not global metrics)

### 🚀 Migration Guide

**Before (v3.5.0 and earlier):**
```bash
curl https://api.example.com/stats
# ✅ 200 OK - Public access
```

**After (v3.5.1+):**
```bash
# Without API key
curl https://api.example.com/stats
# ❌ 401 Unauthorized

# With API key
curl -H "X-Admin-API-Key: YOUR_SECRET_KEY" https://api.example.com/stats
# ✅ 200 OK - Authenticated access
```

**Setup Steps:**
```bash
# 1. Generate secure API key
openssl rand -hex 32

# 2. Set environment variable
export ADMIN_API_KEY="your-generated-key"

# 3. Restart service
docker-compose restart pneumonia-api
```

### 📖 Documentation Updates
- **UPDATED**: `README.md` - Admin endpoints now marked as authenticated
- **UPDATED**: `doc/API_DOCUMENTATION.md` - Authentication requirements documented
- **UPDATED**: `doc/SECURITY-FEATURES.md` - Admin authentication section added
- **ADDED**: `doc/ADMIN_ENDPOINTS_SECURITY.md` - Complete security guide

### ⚠️ Breaking Changes
**Action Required for Existing Users:**
1. Generate and set `ADMIN_API_KEY` environment variable
2. Update monitoring scripts to include `X-Admin-API-Key` header
3. Update CI/CD pipelines that access `/stats` or `/status`
4. Review and update any third-party integrations

**Temporary Workaround (Development Only):**
```bash
# NOT RECOMMENDED FOR PRODUCTION
ENABLE_PUBLIC_STATS=true
ENABLE_PUBLIC_STATUS=true
```

### 🔗 Related Documentation
- [Admin Endpoints Security Guide](doc/ADMIN_ENDPOINTS_SECURITY.md)
- [Security Features Overview](doc/SECURITY-FEATURES.md)
- [API Documentation](doc/API_DOCUMENTATION.md)

---

## [3.5.0] - 2025-09-30 - New Endpoint /badge.json

### ✨ Added
- **New endpoint** `GET /badge.json` in health monitoring:
  - Reuses existing health check logic from health function
  - Provides status badge compatible with shields.io schema
  - Color mapping: 
    - `healthy` → `brightgreen`
    - `partial` → `yellow` 
    - `unhealthy` → `red`
    - `error` → `red`
  - Returns JSON format: `{ "schemaVersion": 1, "label": "API Status", "message": "<status>", "color": "<color>" }`

### Documentation
- Updated README with note about new badge endpoint usage

## [3.4.3] - 2025-09-30 - Render Platform Migration

### ✨ Added
- Introduced official Render deployment configuration via `render.yaml` with automated build and start commands.

### 🔧 Changed
- Default trusted hosts updated from `*.railway.app` to `*.onrender.com` (Railway removed from defaults).
- Deployment script `deploy.sh` updated to reference Render workflow and version bumped to 3.4.3.

### 🗑️ Removed
- Removed `railway.json` (legacy Railway configuration) — migrate to Render Git-based deployment.

### ⚙️ Notes
- Application continues to read dynamic `PORT` env variable (Render sets to 10000 by default for Python services).
- Memory storage remains default; Redis can be re-enabled later without code changes.
- If you still need Railway deployment, restore `railway.json` from git history (tag v3.4.2 or earlier).

---

## [3.4.2] - 2025-09-25 - Comprehensive Architecture Refactoring & Modularization

### 🔥Breaking Changes
- **RESTRUCTURED**: Endpoint response format for `POST /pneumonia/predict`:
  - Fields `model_type` and `model_version` moved to new `model_info` object with `architecture`
  - Legacy consumers parsing root-level fields need updates
- **REPLACED**: Security endpoints restructured:
  - Old unified endpoint replaced with separate endpoints:
    - `GET /security/status` (status & posture)
    - `GET /security/stats` (metrics & aggregation)
- **REMOVED**: Unified Pydantic models in `schemas.py` replaced with modular domain-specific schemas:
  - `prediction_schemas.py`, `health_schemas.py`, `security_schemas.py`, etc.
- **STANDARDIZED**: Error handling format with consistent fields:
  - `error_code`, `message`, `timestamp`, and context-specific attributes (e.g., `retry_after`, `file_hash`)
- **DISABLED**: Redis integration by default (code commented out for manual activation if needed)

### ✨ Added
- **MODULARIZED**: Complete architecture overhaul:
  - Rate limiting system (`app/core/rate_limiting/*`)
  - Dependency injection container (`AppDependencies`)
  - Startup manager (`StartupManager`)
  - OpenAPI documentation builder per endpoint (`app/docs/sections/*`, `app/openapi.py`)
  - Middleware factory with standardized execution order
- **ENHANCED**: Advanced security features:
  - Adaptive attack detection with global threat scoring
  - Fingerprint blocking system
  - Unique IP metrics and behavioral analysis
- **IMPLEMENTED**: Duplicate file detection & in-memory eviction:
  - SHA256 hashing to prevent repeated uploads
  - Memory-efficient cache management
- **RESTRUCTURED**: Prediction response with new `model_info` structure:
  - Contains `type`, `version`, and `architecture` information
- **ADDED**: Development workflow improvements:
  - `.pre-commit-config.yaml` with Black + isort configuration
  - `LADING_PAGE_TEMPLATE.md` for landing page setup
- **INTRODUCED**: `ErrorCode` enum for consistent error handling
- **FIXED**: CORS headers now included in error responses (429, 404, 413) for better browser/Swagger UI compatibility
- **EXPANDED**: Comprehensive internal documentation (refactor guides & architecture docs)

### 🔧 Changed
- **RELAXED**: Default rate limits for better usability:
  - IP limit: increased to 50 requests
  - Fingerprint limit: increased to 10 requests
  - Attack threshold: adjusted to 0.9
- **MODERNIZED**: Application startup using lifecycle context (`lifespan`) with coordinated service initialization
- **ENHANCED**: Structured logging with rich contextual information
- **IMPROVED**: Modular file organization for better maintainability
- **ENRICHED**: Security response data with threat scoring and blocked fingerprint information
- **STANDARDIZED**: Timestamp format to ISO 8601 across all responses

### 🗑️ Removed / Deprecated
- **REMOVED**: `app/models/schemas.py` (replaced with modular schemas)
- **DEPRECATED**: Monolithic rate limiter implementation
- **REPLACED**: Legacy unified security endpoint
- **TEMPORARILY DISABLED**: Active Redis integration (commented out, can be re-enabled)

### 🩹 Fixed
- **RESOLVED**: Inconsistent CORS headers in error responses
- **PREVENTED**: Memory leaks from unmanaged duplicate file hash storage
- **IMPROVED**: Model readiness checking with explicit `is_loaded` validation
- **ENHANCED**: Edge case handling throughout the application

### 🚀 Performance
- **OPTIMIZED**: Modular structure enables selective performance tuning
- **REDUCED**: Duplicate upload processing overhead through intelligent caching
- **IMPROVED**: Container and factory pattern implementation for better resource management

### 🛡️ Security
- **ADVANCED**: Adaptive attack scoring with fingerprint-based blocking
- **ENHANCED**: Security transparency through separate status and statistics endpoints
- **REDUCED**: Default attack surface by disabling external dependencies (Redis)
- **STRENGTHENED**: Request analysis and threat detection capabilities

### 📖 Documentation
- **COMPREHENSIVE**: Production-ready README with security and deployment focus
- **CLEAN**: OpenAPI metadata through dedicated builder system
- **PREPARED**: Landing page template for marketing/presentation needs
- **IMPROVED**: File structure and naming for better developer experience

### ⚙️ Configuration
**New/Changed Environment Variables** (see `settings.py` for full details):
- `RATE_LIMIT_MAX_REQUESTS_PER_IP`: Maximum requests per IP address
- `RATE_LIMIT_MAX_FINGERPRINT_REQUESTS`: Maximum requests per fingerprint
- `ATTACK_THRESHOLD`: Global attack score threshold
- `EXCLUDED_PATHS`: Paths to exclude from rate limiting
- `FINGERPRINT_BLOCK_DURATION_SECONDS`: Duration for fingerprint blocking

**Note**: Adjust your environment configuration to maintain compatibility with previous release behavior if needed.

---

## [3.4.1] - 2025-09-06 - Comprehensive API Documentation Enhancement

### 📚 Documentation Overhaul
- **ENHANCED**: Complete API documentation for all endpoints (/docs and /redoc)
- **IMPROVED**: FastAPI application description with detailed feature overview
- **ADDED**: Comprehensive endpoint documentation with examples and use cases
- **FIXED**: Markdown rendering issues in Swagger UI (switched to plain text)
- **RESOLVED**: Duplicate endpoint display issues in ReDoc interface

### 🔬 Pneumonia Detection Endpoints
- **ENHANCED**: Detailed prediction endpoint documentation with medical context
- **ADDED**: Multi-model support documentation (standard vs EfficientNet-B0)
- **IMPROVED**: Request/response examples with confidence scoring explanations
- **DETAILED**: Medical disclaimer and best practices guidance

### 🏥 Health & Monitoring Documentation
- **COMPREHENSIVE**: Health check endpoint documentation with status interpretations
- **ADDED**: Service uptime and model loading status explanations
- **IMPROVED**: Monitoring use cases and integration guidelines

### 🛡️ Security Documentation
- **DETAILED**: Security system status and statistics endpoint documentation
- **ENHANCED**: Threat level interpretation and protection feature explanations
- **ADDED**: Real-time security metrics and analytics documentation

### 📊 Schema Enhancements
- **IMPROVED**: Pydantic model documentation with detailed field descriptions
- **ADDED**: Response examples and validation patterns
- **ENHANCED**: Medical disclaimer integration in schemas

### 📖 Additional Documentation
- **CREATED**: `API_DOCUMENTATION.md` - Comprehensive API usage guide
- **CREATED**: `USAGE_EXAMPLES.md` - Multi-language client integration examples
- **ADDED**: Best practices, troubleshooting, and error handling guides

### 🔧 Technical Improvements
- **FIXED**: OpenAPI schema configuration for better documentation rendering
- **RESOLVED**: ReDoc duplicate endpoint issues by removing redundant decorators
- **IMPROVED**: FastAPI application metadata and contact information

---

## [3.4.0] - 2025-08-31 - Redis to In-Memory Migration

### 🔄 Architecture Simplification
- **MAJOR**: Migrated from Redis to in-memory storage for rate limiting
  - Changed default storage backend from `redis` to `memory`
  - Made Redis dependency optional for future scalability
  - Implemented graceful fallback mechanisms
- **IMPROVED**: Simplified deployment architecture
  - Removed external Redis dependency requirement
  - Single service deployment for Railway optimization
  - Zero-configuration setup for development

### 🚀 Performance Improvements
- **ENHANCED**: Response time improved by 95% (2-5ms → 0.1ms)
- **OPTIMIZED**: Memory usage reduced by 62.5% (80MB → 30MB)
- **ACCELERATED**: Startup time reduced by 80% (5s → 1s)
- **STREAMLINED**: CPU overhead minimized

### 🛡️ Security & Reliability
- **MAINTAINED**: All advanced rate limiting features preserved
  - IP switching attack detection
  - Request fingerprinting
  - Behavioral analysis
  - Global attack scoring
- **ENHANCED**: Better error handling in storage initialization
- **IMPROVED**: Graceful degradation without external dependencies

### 📦 Dependencies & Deployment
- **SIMPLIFIED**: Reduced dependencies by 33% (15 → 10 packages)
- **OPTIMIZED**: Railway deployment compatibility improved
- **ENHANCED**: Docker deployment simplified (single container)
- **IMPROVED**: Development experience (no external setup)

### 📚 Documentation
- **ADDED**: `IN_MEMORY_MIGRATION_GUIDE.md` - Complete migration documentation
- **ADDED**: `MIGRATION_TEST_RESULTS.md` - Test results and verification
- **UPDATED**: All documentation to reflect in-memory storage usage
- **ENHANCED**: Deployment guides for simplified architecture

### 🔧 Code Quality
- **REFACTORED**: Storage factory with optional Redis imports
- **IMPROVED**: Error handling in application lifecycle
- **ENHANCED**: Global variable management for rate limiter
- **FIXED**: Shutdown process for in-memory storage

---

## [3.3.1] - 2025-08-25 - Code Quality & Security Improvements

### 🔧 Code Refactoring
- **IMPROVED**: Refactored `PneumoniaPredictionService` with extracted helper methods
  - Added `_extract_model_type()` method for safer model type detection
  - Added `_get_target_size()` method for cleaner preprocessing logic
  - Enhanced error handling with proper fallback mechanisms
- **IMPROVED**: Restructured `SecurityMiddleware` for better maintainability
  - Extracted `_should_skip_rate_limiting()` for endpoint filtering
  - Added `_extract_file_hash()` for cleaner file handling
  - Created `_check_rate_limit()` for modular rate limiting logic
  - Implemented `_create_rate_limit_response()` for consistent error responses

### 🛡️ Security Enhancements
- **ENHANCED**: Better rate limiting fallback handling
- **IMPROVED**: More detailed error responses with security headers
- **FIXED**: Proper exception propagation in middleware chain
- **ADDED**: Enhanced logging for security events

### 🔬 Testing & Quality
- **IMPROVED**: Better error handling in all components
- **ENHANCED**: Code readability and maintainability
- **FIXED**: Potential edge cases in model type detection
- **IMPROVED**: Middleware error recovery mechanisms

---

## [3.3.0] - 2025-09-01 - Railway Production Optimization

### 🚀 Railway Deployment Fixes
- **FIXED**: Redis configuration issues in Railway production environment
- **CHANGED**: Default storage backend from Redis to Memory for Railway
- **IMPROVED**: Rate limiter initialization with better fallback mechanisms
- **ADDED**: Basic rate limiting as fallback when advanced system fails

### 🛡️ Enhanced Production Stability
- **IMPROVED**: Graceful degradation when rate limiter components fail
- **ENHANCED**: Startup error handling and logging for better debugging
- **FIXED**: Memory storage configuration for Railway deployment
- **ADDED**: Basic time-based rate limiting as ultimate fallback

### 📝 Configuration Updates
- **UPDATED**: railway.json to use memory storage instead of Redis
- **FIXED**: Production environment variables for Railway
- **IMPROVED**: Health check reliability in production
- **OPTIMIZED**: Docker configuration for Railway deployment

### 🔧 Technical Improvements
- **ENHANCED**: SecurityMiddleware with basic rate limiting fallback
- **IMPROVED**: Rate limiter initialization sequence
- **FIXED**: Production warnings about rate limiter not initialized
- **ADDED**: Better error messages and logging for troubleshooting

---

## [3.1.0] - 2025-08-26 - Health Check & Production Fixes

### 🚑 Health Check Improvements
- **FIXED**: Railway deployment health check failures
- **ADDED**: /health endpoint alias for Railway compatibility
- **IMPROVED**: Startup resilience with graceful fallbacks
- **ENHANCED**: Production logging and error handling

---

## [2.1.0] - 2025-08-25 - Architecture Overhaul & Code Quality

### 🏗️ Architecture Overhaul
- **BREAKING**: Complete restructure from monolithic `main.py` to modular architecture
- **NEW**: Clean separation of concerns across layers (API, Services, Utils, Core)
- **NEW**: Dependency injection pattern for better testability
- **NEW**: Configuration management with environment variable support

### 📁 New Project Structure
```
app/
├── core/           # Application core (settings, logging)
├── api/            # API route handlers  
├── services/       # Business logic layer
├── models/         # Pydantic data models
├── utils/          # Utility functions
└── middleware/     # Custom middleware
```

### ✨ New Features
- **Configuration Management**: Centralized settings with `.env` support
- **Enhanced Logging**: Structured logging with configurable levels
- **Custom Exceptions**: Proper exception hierarchy for different error types
- **Security Middleware**: Request logging and rate limiting middleware
- **Model Information Endpoint**: New `/pneumonia/model/info` endpoint
- **Improved Validation**: Enhanced image content validation for medical relevance

### 🔒 Security Improvements
- **Rate Limiting**: Enhanced rate limiter with IP blocking and configurable limits
- **File Validation**: Improved file size, type, and content validation
- **Security Headers**: Proper CORS and trusted host configuration
- **Error Handling**: Secure error responses without information leakage

### 🛠️ Developer Experience
- **Auto-generated Documentation**: Enhanced OpenAPI/Swagger documentation
- **Type Safety**: Full type hints throughout the codebase
- **Code Quality**: Clean code principles with proper formatting
- **Modular Design**: Easy to test, maintain, and extend

### 📊 Monitoring & Observability
- **Health Checks**: Comprehensive health check with uptime tracking
- **Request Logging**: Detailed request/response logging with performance metrics
- **Security Monitoring**: Security status endpoint for monitoring

### 🔄 Backward Compatibility
- **Legacy Support**: `main.py` still works as entry point for backward compatibility
- **Same API Endpoints**: All existing endpoints maintained with same functionality
- **Environment Variables**: Existing environment variables still supported

### 📝 Documentation
- **Architecture Documentation**: Comprehensive `ARCHITECTURE.md` file
- **Updated README**: Improved documentation with new features
- **Code Comments**: Enhanced inline documentation

### 🐛 Bug Fixes
- **Memory Management**: Better memory handling for image processing
- **Error Handling**: More robust error handling throughout the application
- **Configuration**: Fixed configuration loading issues

### ⚡ Performance Improvements
- **Async Operations**: Better async handling for file operations
- **Memory Efficiency**: Optimized image processing pipeline
- **Startup Time**: Faster application startup with lazy loading

### 🧪 Testing
- **Test Structure**: Foundation for comprehensive testing with dependency injection
- **Mock Support**: Easy mocking of services for unit testing
- **Integration Testing**: Structure supports integration testing

---

## [1.0.0] - Previous Version

### Features
- Basic pneumonia detection API
- ONNX model integration
- File upload validation
- Rate limiting
- Basic security measures
