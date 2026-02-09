"""Graceful shutdown support for experiment runners.

This module provides thread-safe tracking of shutdown state and active environments,
enabling graceful cleanup when experiments are interrupted (SIGINT/SIGTERM).

Usage:
    # In main script:
    from agentic_uncertainty.scripts._shared.shutdown import (
        is_shutdown_requested, request_shutdown, cleanup_all_environments,
        register_environment, unregister_environment
    )
    
    # In elicitation code:
    from agentic_uncertainty.scripts._shared.shutdown import (
        register_environment, unregister_environment
    )
    
    env = create_environment()
    register_environment(env)
    try:
        # ... use environment ...
    finally:
        unregister_environment(env)
        env.stop()
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Global tracking for graceful shutdown
_shutdown_requested = False
_shutdown_lock = threading.Lock()
_active_environments: set = set()  # Track all running environments


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested.
    
    Returns:
        True if shutdown was requested via signal or request_shutdown().
    """
    return _shutdown_requested


def request_shutdown() -> None:
    """Request graceful shutdown.
    
    This sets the shutdown flag, which can be checked by running tasks
    to stop processing new work.
    """
    global _shutdown_requested
    with _shutdown_lock:
        _shutdown_requested = True


def reset_shutdown() -> None:
    """Reset shutdown state (for testing purposes)."""
    global _shutdown_requested
    with _shutdown_lock:
        _shutdown_requested = False
        _active_environments.clear()


def register_environment(env: Any) -> None:
    """Register an environment for cleanup tracking.
    
    Call this immediately after creating an environment. The environment
    will be cleaned up on shutdown if not explicitly unregistered.
    
    Args:
        env: Environment instance with stop() or cleanup() method.
    """
    with _shutdown_lock:
        _active_environments.add(env)
        logger.debug(f"Registered environment: {type(env).__name__} (total: {len(_active_environments)})")


def unregister_environment(env: Any) -> None:
    """Unregister an environment from cleanup tracking.
    
    Call this after successfully cleaning up an environment to prevent
    double-cleanup on shutdown.
    
    Args:
        env: Environment instance to remove.
    """
    with _shutdown_lock:
        _active_environments.discard(env)
        logger.debug(f"Unregistered environment: {type(env).__name__} (total: {len(_active_environments)})")


def get_active_environment_count() -> int:
    """Get the number of currently active environments.
    
    Returns:
        Number of registered environments.
    """
    with _shutdown_lock:
        return len(_active_environments)


def cleanup_all_environments() -> int:
    """Stop all registered environments.
    
    This is called during shutdown to ensure all sandboxes (Modal, Docker, etc.)
    are properly terminated.
    
    Returns:
        Number of environments successfully cleaned up.
    """
    with _shutdown_lock:
        envs_to_cleanup = list(_active_environments)
    
    if not envs_to_cleanup:
        return 0
    
    logger.info(f"Cleaning up {len(envs_to_cleanup)} environment(s)...")
    cleaned = 0
    
    for env in envs_to_cleanup:
        env_type = type(env).__name__
        try:
            if hasattr(env, 'stop'):
                env.stop()
                logger.debug(f"Stopped environment: {env_type}")
            elif hasattr(env, 'cleanup'):
                env.cleanup()
                logger.debug(f"Cleaned up environment: {env_type}")
            cleaned += 1
        except Exception as e:
            logger.warning(f"Error during {env_type} cleanup: {e}")
    
    with _shutdown_lock:
        _active_environments.clear()
    
    logger.info(f"Cleaned up {cleaned}/{len(envs_to_cleanup)} environment(s)")
    return cleaned


def atexit_cleanup() -> None:
    """Cleanup handler for use with atexit module.
    
    This provides a fallback cleanup mechanism. Register it via:
        import atexit
        atexit.register(atexit_cleanup)
    """
    global _shutdown_requested
    _shutdown_requested = True
    
    n_envs = get_active_environment_count()
    if n_envs > 0:
        # Use print since logging may not work in atexit
        print(f"\nCleaning up {n_envs} environment(s)...", file=sys.stderr)
        cleaned = cleanup_all_environments()
        if cleaned > 0:
            print(f"Cleaned up {cleaned} environment(s).", file=sys.stderr)
