"""Modal configuration - ensures .env is loaded before Modal is used."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing modal to ensure MODAL_PROFILE is set
_env_loaded = False


def ensure_env_loaded() -> None:
    """Ensure environment variables are loaded from .env file."""
    global _env_loaded
    if _env_loaded:
        return

    # Try to find .env in current dir or parent dirs
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try the package root
        package_root = Path(__file__).parent.parent.parent.parent
        env_path = package_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)

    _env_loaded = True


def get_modal():
    """Get the modal module with environment properly configured.

    Returns:
        The modal module, or None if not installed.
    """
    ensure_env_loaded()

    try:
        import modal
        return modal
    except ImportError:
        return None


def get_modal_profile() -> str | None:
    """Get the configured Modal profile from environment."""
    ensure_env_loaded()
    return os.getenv("MODAL_PROFILE")


def get_modal_app(name: str = "agentic-uncertainty"):
    """Get or create a Modal app with the correct profile.

    Args:
        name: Name for the Modal app.

    Returns:
        A Modal App instance.
    """
    modal = get_modal()
    if modal is None:
        raise RuntimeError("Modal is not installed. Run: uv add modal")

    return modal.App.lookup(name=name, create_if_missing=True)
