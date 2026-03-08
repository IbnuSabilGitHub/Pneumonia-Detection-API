"""
ES256 JWT Authentication for Supabase Integration

Provides JWT token verification, user extraction, and role-based
access control using Supabase as the authentication provider.

Features:
- ES256 (Elliptic Curve): Signs tokens with Supabase private key
- Verifies using public key fetched from Supabase JWKS endpoint
- Asymmetric: No shared secret required on API side
"""

from typing import Optional
import json
import requests
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.logger import get_logger
from ..core.settings import settings

logger = get_logger(__name__)

# Bearer token scheme for OpenAPI docs
bearer_scheme = HTTPBearer(
    scheme_name="Supabase JWT",
    description="JWT token from Supabase authentication. "
    "Pass the `access_token` returned by Supabase sign-in.",
    auto_error=False,
)


class JWTPayload:
    """
    Parsed JWT payload with helper properties.

    Attributes:
        sub: Subject (Supabase user ID).
        email: User email address.
        role: Supabase role (e.g. ``authenticated``, ``service_role``).
        raw: Complete decoded payload dictionary.
    """

    def __init__(self, payload: dict):
        self.sub: str = payload.get("sub", "")
        self.email: Optional[str] = payload.get("email")
        self.role: str = payload.get("role", "authenticated")
        self.raw: dict = payload

    @property
    def user_id(self) -> str:
        """Alias for ``sub`` — the Supabase user ID."""
        return self.sub

    @property
    def is_admin(self) -> bool:
        """Check whether the token carries an admin / service role."""
        admin_roles = {"service_role", "admin"}
        # Check top-level role
        if self.role in admin_roles:
            return True
        # Check app_metadata.role (Supabase custom claims)
        app_meta = self.raw.get("app_metadata", {})
        if app_meta.get("role") in admin_roles:
            return True
        # Check user_metadata.role
        user_meta = self.raw.get("user_metadata", {})
        if user_meta.get("role") in admin_roles:
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"JWTPayload(sub={self.sub!r}, email={self.email!r}, "
            f"role={self.role!r}, is_admin={self.is_admin})"
        )


@lru_cache(maxsize=1)
def _get_supabase_jwks() -> dict:
    """
    Fetch Supabase JWKS (JSON Web Key Set) public keys.
    
    This is used to verify ES256 tokens from Supabase production.
    Results are cached since JWKS changes infrequently.
    
    Returns:
        JWKS dictionary with keys.
        
    Raises:
        HTTPException: If JWKS cannot be fetched.
    """
    if not settings.supabase_url:
        logger.error("SUPABASE_URL not configured - cannot fetch JWKS")
        return {}
    
    try:
        jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
        response = requests.get(jwks_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("Failed to fetch Supabase JWKS: %s", exc)
        return {}


def _get_public_key_for_token(token: str) -> Optional[object]:
    """
    Extract ES256 public key from token header and Supabase JWKS.
    
    Args:
        token: JWT token string.
        
    Returns:
        EC public key object for token verification, or None if not found.
    """
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")  # Key ID from token header
        
        if not kid:
            logger.debug("No key ID (kid) found in token header")
            return None
        
        jwks = _get_supabase_jwks()
        if not jwks or "keys" not in jwks:
            logger.error("JWKS not available")
            return None
        
        # Find the key matching the kid
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == kid:
                # Only ES256 (ECDSA) keys are supported
                if key_data.get("alg") != "ES256":
                    logger.error("Unexpected algorithm in JWKS: %s (expected ES256)", key_data.get("alg"))
                    return None
                    
                from jwt.algorithms import ECAlgorithm
                return ECAlgorithm.from_jwk(json.dumps(key_data))
        
        logger.error("Key with kid %s not found in Supabase JWKS", kid)
        return None
        
    except Exception as exc:
        logger.error("Error extracting public key from token: %s", exc)
        return None


def _decode_token(token: str) -> dict:
    """
    Decode and verify an ES256 JWT token from Supabase.
    
    Uses Supabase public key (fetched from JWKS endpoint) to verify
    the cryptographic signature. No shared secret required.

    Args:
        token: Raw JWT string.

    Returns:
        Decoded payload dictionary.

    Raises:
        HTTPException: On any verification failure.
    """
    if not settings.supabase_url:
        logger.error("SUPABASE_URL not configured - cannot verify ES256 tokens")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Authentication service not configured",
                "error_code": "AUTH_NOT_CONFIGURED",
                "message": "Supabase URL not configured. Contact administrator.",
            },
        )

    try:
        # Get ES256 public key from Supabase JWKS
        logger.debug("Verifying ES256 token with Supabase JWKS")
        public_key = _get_public_key_for_token(token)
        
        if not public_key:
            logger.error("Failed to extract public key from token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "Invalid token",
                    "error_code": "INVALID_TOKEN",
                    "message": "Token signature verification failed. Ensure you are using a valid Supabase access token.",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify ES256 signature
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["ES256"],
            audience="authenticated",
            options={
                "verify_aud": bool(settings.supabase_jwt_verify_audience),
                "verify_exp": True,
                "verify_iat": False,  # Skip iat check due to potential clock skew
                "require": ["sub", "exp"],
            },
        )
        logger.debug("ES256 token verified successfully for user %s", payload.get("sub"))
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Token expired",
                "error_code": "TOKEN_EXPIRED",
                "message": "Your session has expired. Please sign in again.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        logger.warning("JWT audience mismatch")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid token audience",
                "error_code": "INVALID_AUDIENCE",
                "message": "Token was not issued for this service.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.DecodeError as exc:
        logger.warning("JWT decode error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid token",
                "error_code": "INVALID_TOKEN",
                "message": "The provided token is malformed or invalid.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        logger.warning("JWT validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Token validation failed",
                "error_code": "TOKEN_VALIDATION_FAILED",
                "message": "Token signature or format is invalid. Ensure you're using a valid Supabase access token.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> JWTPayload:
    """
    FastAPI dependency — extract and verify the authenticated user.

    Usage::

        @router.post("/protected")
        async def protected_route(user: JWTPayload = Depends(get_current_user)):
            ...

    Raises:
        HTTPException 401: Missing or invalid token.
    """
    if not settings.jwt_auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "JWT authentication disabled",
                "error_code": "AUTH_DISABLED",
                "message": "JWT authentication is not enabled on this instance.",
            },
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Missing authentication",
                "error_code": "MISSING_TOKEN",
                "message": "Authorization header with Bearer token is required.",
                "hint": "Add header: Authorization: Bearer <your_supabase_access_token>",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_token(credentials.credentials)
    user = JWTPayload(payload)
    logger.debug("Authenticated user: %s (role=%s)", user.user_id, user.role)
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[JWTPayload]:
    """
    FastAPI dependency — optionally extract user if token is provided.

    Returns ``None`` when no token is present (instead of raising 401).
    Useful for endpoints that behave differently for authenticated users.
    """
    if not settings.jwt_auth_enabled or credentials is None:
        return None

    payload = _decode_token(credentials.credentials)
    return JWTPayload(payload)


async def get_admin_user(
    user: JWTPayload = Depends(get_current_user),
) -> JWTPayload:
    """
    FastAPI dependency — require an admin-level JWT user.

    Checks ``role``, ``app_metadata.role``, and ``user_metadata.role``
    for ``admin`` or ``service_role``.

    Raises:
        HTTPException 403: When the user lacks admin privileges.
    """
    if not user.is_admin:
        logger.warning(
            "Non-admin user %s attempted admin access (role=%s)",
            user.user_id,
            user.role,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Insufficient permissions",
                "error_code": "ADMIN_REQUIRED",
                "message": "This endpoint requires admin privileges.",
                "current_role": user.role,
            },
        )
    logger.info("Admin access granted for user %s", user.user_id)
    return user


async def verify_admin_jwt_or_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """
    FastAPI dependency — accept **either** a Supabase JWT (admin role)
    **or** the legacy ``X-Admin-API-Key`` header.

    This allows a smooth migration while keeping backward compatibility
    with the existing API-key-based admin authentication.

    Returns:
        str: Identifier of the authenticated admin
             (user-id for JWT, ``"api_key"`` for legacy key).
    """
    # --- Attempt 1: JWT Bearer token ---
    if settings.jwt_auth_enabled and credentials is not None:
        try:
            payload = _decode_token(credentials.credentials)
            user = JWTPayload(payload)
            if not user.is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "Insufficient permissions",
                        "error_code": "ADMIN_REQUIRED",
                        "message": "Admin JWT role required for this endpoint.",
                        "current_role": user.role,
                    },
                )
            logger.info("Admin JWT access granted for user %s", user.user_id)
            return user.user_id
        except HTTPException:
            raise  # re-raise auth errors as-is

    # --- Attempt 2: Legacy API Key header ---
    from .auth import verify_admin_api_key as _legacy_verify  # avoid circular imports

    api_key_header_value = request.headers.get("X-Admin-API-Key")
    if api_key_header_value:
        await _legacy_verify(api_key=api_key_header_value)
        return "api_key"

    # --- Neither provided ---
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": "Authentication required",
            "error_code": "MISSING_CREDENTIALS",
            "message": (
                "Provide a Supabase JWT Bearer token "
                "or X-Admin-API-Key header."
            ),
        },
        headers={"WWW-Authenticate": "Bearer"},
    )

