"""
Redis storage backend implementation for advanced rate limiting.
Production-ready with clustering, connection pooling, and error handling.
"""

import json
import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from .storage_backends import StorageBackend

logger = logging.getLogger(__name__)


class RedisStorageBackend(StorageBackend):
    """Redis storage backend with clustering and error handling."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        max_connections: int = 50,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        cluster_mode: bool = False,
        cluster_nodes: Optional[List[Dict]] = None,
        key_prefix: str = "rate_limit:",
        encoding: str = "utf-8",
    ):
        """Initialize Redis storage backend."""
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.max_connections = max_connections
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval
        self.cluster_mode = cluster_mode
        self.cluster_nodes = cluster_nodes or []
        self.key_prefix = key_prefix
        self.encoding = encoding

        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self.last_health_check = 0
        self.is_healthy = False

        # Script hashes for Lua scripts
        self.script_hashes = {}

    async def connect(self) -> bool:
        """Initialize Redis connection."""
        try:
            if self.cluster_mode:
                # Redis Cluster mode
                startup_nodes = self.cluster_nodes or [
                    {"host": self.host, "port": self.port}
                ]
                from redis.asyncio.cluster import RedisCluster

                self.redis_client = RedisCluster(
                    startup_nodes=startup_nodes,
                    password=self.password,
                    decode_responses=True,
                    skip_full_coverage_check=True,
                    max_connections_per_node=self.max_connections // len(startup_nodes),
                )
            else:
                # Single Redis instance
                self.connection_pool = ConnectionPool(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    db=self.db,
                    max_connections=self.max_connections,
                    retry_on_timeout=self.retry_on_timeout,
                    decode_responses=True,
                )
                self.redis_client = redis.Redis(connection_pool=self.connection_pool)

            # Test connection
            await self.redis_client.ping()
            self.is_healthy = True
            self.last_health_check = time.time()

            # Load Lua scripts
            await self._load_lua_scripts()

            logger.info("Redis connection established: %s:%d", self.host, self.port)
            return True

        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            self.is_healthy = False
            return False

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        if self.connection_pool:
            await self.connection_pool.disconnect()
            self.connection_pool = None
        logger.info("Redis connection closed")

    async def _load_lua_scripts(self):
        """Load Lua scripts for atomic operations."""
        # Atomic increment with TTL
        increment_script = """
        local key = KEYS[1]
        local increment = tonumber(ARGV[1])
        local ttl = tonumber(ARGV[2])

        local current = redis.call('GET', key)
        if current == false then
            current = 0
        else
            current = tonumber(current)
        end

        local new_value = current + increment
        redis.call('SET', key, new_value)

        if ttl > 0 then
            redis.call('EXPIRE', key, ttl)
        end

        return new_value
        """

        # List append with max length
        list_append_script = """
        local key = KEYS[1]
        local value = ARGV[1]
        local max_length = tonumber(ARGV[2])

        redis.call('LPUSH', key, value)

        if max_length > 0 then
            redis.call('LTRIM', key, 0, max_length - 1)
        end

        return redis.call('LLEN', key)
        """

        try:
            self.script_hashes["increment"] = await self.redis_client.script_load(
                increment_script
            )
            self.script_hashes["list_append"] = await self.redis_client.script_load(
                list_append_script
            )
        except Exception as e:
            logger.warning("Failed to load Lua scripts: %s", e)

    def _make_key(self, key: str) -> str:
        """Create prefixed Redis key."""
        return f"{self.key_prefix}{key}"

    async def _ensure_connection(self) -> bool:
        """Ensure Redis connection is healthy."""
        current_time = time.time()

        # Check health periodically
        if current_time - self.last_health_check > self.health_check_interval:
            try:
                if self.redis_client:
                    await self.redis_client.ping()
                    self.is_healthy = True
                else:
                    self.is_healthy = False
                self.last_health_check = current_time
            except Exception as e:
                logger.error("Redis health check failed: %s", e)
                self.is_healthy = False
                # Try to reconnect
                await self.connect()

        return self.is_healthy

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        if not await self._ensure_connection():
            return None

        try:
            redis_key = self._make_key(key)
            value = await self.redis_client.get(redis_key)

            if value is None:
                return None

            # Try to parse as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as e:
            logger.error("Redis GET error for key %s: %s", key, e)
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        if not await self._ensure_connection():
            return False

        try:
            redis_key = self._make_key(key)

            # Serialize complex objects to JSON
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, default=str)

            if ttl:
                await self.redis_client.setex(redis_key, ttl, value)
            else:
                await self.redis_client.set(redis_key, value)

            return True

        except Exception as e:
            logger.error("Redis SET error for key %s: %s", key, e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete key."""
        if not await self._ensure_connection():
            return False

        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            return result > 0

        except Exception as e:
            logger.error("Redis DELETE error for key %s: %s", key, e)
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not await self._ensure_connection():
            return False

        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.exists(redis_key)
            return result > 0

        except Exception as e:
            logger.error("Redis EXISTS error for key %s: %s", key, e)
            return False

    async def increment(
        self, key: str, amount: int = 1, ttl: Optional[int] = None
    ) -> int:
        """Increment counter with atomic operation."""
        if not await self._ensure_connection():
            return 0

        try:
            redis_key = self._make_key(key)

            # Use Lua script for atomic increment with TTL
            if "increment" in self.script_hashes:
                result = await self.redis_client.evalsha(
                    self.script_hashes["increment"], 1, redis_key, amount, ttl or 0
                )
                return int(result)
            else:
                # Fallback to basic increment
                result = await self.redis_client.incrby(redis_key, amount)
                if ttl:
                    await self.redis_client.expire(redis_key, ttl)
                return result

        except Exception as e:
            logger.error("Redis INCREMENT error for key %s: %s", key, e)
            return 0

    async def get_list(self, key: str) -> List[Any]:
        """Get list values."""
        if not await self._ensure_connection():
            return []

        try:
            redis_key = self._make_key(key)
            values = await self.redis_client.lrange(redis_key, 0, -1)

            # Parse JSON values
            result = []
            for value in values:
                try:
                    result.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    result.append(value)

            return result

        except Exception as e:
            logger.error("Redis GET_LIST error for key %s: %s", key, e)
            return []

    async def append_to_list(
        self, key: str, value: Any, max_length: Optional[int] = None
    ) -> bool:
        """Append to list with optional max length."""
        if not await self._ensure_connection():
            return False

        try:
            redis_key = self._make_key(key)

            # Serialize value
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, default=str)

            # Use Lua script for atomic append with trimming
            if "list_append" in self.script_hashes and max_length:
                await self.redis_client.evalsha(
                    self.script_hashes["list_append"], 1, redis_key, value, max_length
                )
            else:
                # Basic append
                await self.redis_client.lpush(redis_key, value)
                if max_length:
                    await self.redis_client.ltrim(redis_key, 0, max_length - 1)

            return True

        except Exception as e:
            logger.error("Redis APPEND_TO_LIST error for key %s: %s", key, e)
            return False

    async def get_hash(self, key: str) -> Dict[str, Any]:
        """Get hash/dict values."""
        if not await self._ensure_connection():
            return {}

        try:
            redis_key = self._make_key(key)
            hash_data = await self.redis_client.hgetall(redis_key)

            # Parse JSON values
            result = {}
            for field, value in hash_data.items():
                try:
                    result[field] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[field] = value

            return result

        except Exception as e:
            logger.error("Redis GET_HASH error for key %s: %s", key, e)
            return {}

    async def set_hash_field(self, key: str, field: str, value: Any) -> bool:
        """Set hash field."""
        if not await self._ensure_connection():
            return False

        try:
            redis_key = self._make_key(key)

            # Serialize complex objects
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, default=str)

            await self.redis_client.hset(redis_key, field, value)
            return True

        except Exception as e:
            logger.error(
                f"Redis SET_HASH_FIELD error for key {key}, field {field}: {e}"
            )
            return False

    async def cleanup_expired(self) -> int:
        """Clean up expired entries (Redis handles this automatically)."""
        # Redis automatically handles TTL expiration
        # We can return an estimate based on memory usage
        if not await self._ensure_connection():
            return 0

        try:
            info = await self.redis_client.info("memory")
            expired_keys = info.get("expired_keys", 0)
            return expired_keys
        except Exception as e:
            logger.error("Redis CLEANUP_EXPIRED error: %s", e)
            return 0

    async def ping(self) -> bool:
        """Test connection."""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                return True
        except Exception as e:
            logger.error("Redis PING error: %s", e)
        return False

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        info = {
            "backend_type": "redis",
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "cluster_mode": self.cluster_mode,
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check,
            "key_prefix": self.key_prefix,
        }

        if await self._ensure_connection():
            try:
                redis_info = await self.redis_client.info()
                info.update(
                    {
                        "redis_version": redis_info.get("redis_version"),
                        "used_memory": redis_info.get("used_memory_human"),
                        "connected_clients": redis_info.get("connected_clients"),
                        "total_commands_processed": redis_info.get(
                            "total_commands_processed"
                        ),
                        "keyspace_hits": redis_info.get("keyspace_hits"),
                        "keyspace_misses": redis_info.get("keyspace_misses"),
                        "uptime_in_seconds": redis_info.get("uptime_in_seconds"),
                    }
                )
            except Exception as e:
                logger.error("Failed to get Redis info: %s", e)

        return info
