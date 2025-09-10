# Advanced Rate Limiting Refactor

This refactor consolidates the advanced rate limiting implementation into a dedicated package `app.core.rate_limiting` and keeps `app.core.advanced_rate_limiting` as a thin compatibility layer.

## What Changed

- New package: `app/core/rate_limiting/`
  - `core.py`: AdvancedRateLimiter implementation
  - `api.py`: factory `create_advanced_rate_limiter`
  - `storage.py`: dataclasses and storage models
- Old module `app/core/advanced_rate_limiting.py` now re-exports from the new package and keeps the global getter/setter. It logs a debug deprecation hint.

## Why

- Clear separation of concerns and better maintainability
- Easier testing and future extensions
- Backward compatible for existing imports

## How to Import (Recommended)

- Preferred
  - from app.core.rate_limiting.core import AdvancedRateLimiter
  - from app.core.rate_limiting.api import create_advanced_rate_limiter

- Backward compatible (still works)
  - from app.core.advanced_rate_limiting import AdvancedRateLimiter, create_advanced_rate_limiter

## Defaults and Thresholds

Current defaults (as of this refactor):
- window_size = 60
- max_requests_per_ip = 10
- max_fingerprint_requests = 3
- suspicious_ip_changes_threshold = 5
- attack_block_duration = 300
- ip_switching_threshold = 3
- global_attack_threshold = 0.8
- bot_behavior_variance = 0.1

## Notes

- The diagrams in `doc/ADVANCED_RATE_LIMITING_DOCS.md` are illustrative. Where they show different numeric examples, prefer the values above.
- Security middleware already integrates with the global getter/setter.
- No public API was removed; imports continue to work.
