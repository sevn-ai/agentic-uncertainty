import logging
import os
import threading
from pathlib import Path
from typing import Callable

from rich.logging import RichHandler
from rich.text import Text

_SET_UP_LOGGERS: set[str] = set()
_ADDITIONAL_HANDLERS: dict[str, logging.Handler] = {}
_LOG_LOCK = threading.Lock()

logging.TRACE = 5  # type: ignore
logging.addLevelName(logging.TRACE, "TRACE")  # type: ignore


def _interpret_level(level: int | str | None, *, default=logging.DEBUG) -> int:
    if not level:
        return default
    if isinstance(level, int):
        return level
    if level.isnumeric():
        return int(level)
    return getattr(logging, level.upper())


_STREAM_LEVEL = _interpret_level(os.environ.get("MSWE_AGENT_LOG_STREAM_LEVEL"))
_INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER = False

_THREAD_NAME_TO_LOG_SUFFIX: dict[str, str] = {}
"""Mapping from thread name to suffix to add to the logger name."""


class _RichHandlerWithEmoji(RichHandler):
    """A RichHandler that prepends an emoji to the level name."""

    def __init__(self, *args, emoji: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.emoji = emoji

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        return Text.styled(
            (self.emoji + level_name).ljust(10), f"logging.level.{level_name.lower()}"
        )


def _setup_root_logger() -> None:
    logger = logging.getLogger("minisweagent")
    logger.setLevel(logging.DEBUG)
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


def add_file_handler(
    path: Path | str,
    *,
    filter: str | Callable[[str], bool] | None = None,
    level: int | str = logging.DEBUG,
    id_: str = "",
    print_path: bool = True,
) -> str:
    """Add a file handler to all loggers.

    Args:
        path: Path to the log file
        filter: If str: Check that the logger name contains the filter string.
            If callable: Check that the logger name satisfies the condition.
        level: The level of the handler
        id_: The id of the handler. If not provided, a random id will be generated.
        print_path: Whether to print the path to the log file

    Returns:
        The id of the handler. Can be used to remove the handler later.
    """
    import uuid

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    handler.setLevel(_interpret_level(level))

    with _LOG_LOCK:
        for name in _SET_UP_LOGGERS:
            if filter is not None:
                if isinstance(filter, str) and filter not in name:
                    continue
                if callable(filter) and not filter(name):
                    continue
            logger = logging.getLogger(name)
            logger.addHandler(handler)

    handler.my_filter = filter  # type: ignore
    if not id_:
        id_ = str(uuid.uuid4())
    _ADDITIONAL_HANDLERS[id_] = handler

    if print_path:
        print(f"Logging to '{path}'")

    return id_


def remove_file_handler(id_: str) -> None:
    """Remove a file handler by its id."""
    if id_ not in _ADDITIONAL_HANDLERS:
        return
    handler = _ADDITIONAL_HANDLERS.pop(id_)
    with _LOG_LOCK:
        for log_name in _SET_UP_LOGGERS:
            logger = logging.getLogger(log_name)
            logger.removeHandler(handler)


def get_logger(name: str, *, emoji: str = "") -> logging.Logger:
    """Get logger. Use this instead of `logging.getLogger` to ensure
    that the logger is set up with the correct handlers.
    """
    thread_name = threading.current_thread().name
    if thread_name != "MainThread":
        name = name + "-" + _THREAD_NAME_TO_LOG_SUFFIX.get(thread_name, thread_name)
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Already set up
        return logger
    handler = _RichHandlerWithEmoji(
        emoji=emoji,
        show_time=bool(os.environ.get("MSWE_AGENT_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)
    # Set to lowest level and only use stream handlers to adjust levels
    logger.setLevel(logging.TRACE)  # type: ignore
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)
    with _LOG_LOCK:
        for handler in _ADDITIONAL_HANDLERS.values():
            my_filter = getattr(handler, "my_filter", None)
            if my_filter is None:
                logger.addHandler(handler)
            elif isinstance(my_filter, str) and my_filter in name:
                logger.addHandler(handler)
            elif callable(my_filter) and my_filter(name):
                logger.addHandler(handler)
    if _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER:
        _add_logger_name_to_stream_handler(logger)
    return logger


_setup_root_logger()
logger = logging.getLogger("minisweagent")


__all__ = ["logger", "add_file_handler", "remove_file_handler", "get_logger"]
