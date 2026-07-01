"""Colored stderr logging for CLI and demos."""

from __future__ import annotations

import logging
import sys

_RESET = "\033[0m"
_LEVEL_STYLES: dict[int, tuple[str, str]] = {
    logging.DEBUG: ("\033[36m", "DBG"),
    logging.INFO: ("\033[32m", "INF"),
    logging.WARNING: ("\033[33m", "WRN"),
    logging.ERROR: ("\033[31m", "ERR"),
    logging.CRITICAL: ("\033[35m", "CRT"),
}


class _ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color, label = _LEVEL_STYLES.get(record.levelno, ("\033[37m", "???"))
        record.levelname = f"{color}{label}{_RESET}"
        record.name = f"\033[90m{record.name}{_RESET}"
        return super().format(record)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure package logger with colored stderr output."""
    logger = logging.getLogger("convolutional_wasserstein")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColoredFormatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
