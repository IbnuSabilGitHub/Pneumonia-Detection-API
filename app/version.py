"""Central API version definition.

Update ``API_VERSION`` here to propagate the default version across
application metadata, OpenAPI docs, and fallback entrypoint responses.
"""

import os
from pathlib import Path

API_VERSION = "3.7.1"


def _resolve_current_api_version() -> str:
	"""Resolve API version from environment, then .env, then code default."""
	# Priority 1: real environment variable
	env_version = os.getenv("APP_VERSION")
	if env_version:
		return env_version

	# Priority 2: .env file value (for local development fallback paths)
	try:
		from dotenv import dotenv_values

		env_path = Path(__file__).resolve().parent.parent / ".env"
		env_data = dotenv_values(env_path)
		file_version = env_data.get("APP_VERSION")
		if file_version:
			return str(file_version)
	except Exception:
		# Ignore dotenv read errors and use code default.
		pass

	# Priority 3: central code default
	return API_VERSION


CURRENT_API_VERSION = _resolve_current_api_version()
