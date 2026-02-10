"""Veritas classifier pipeline — utilities."""

import logging
import time
from contextlib import contextmanager


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the pipeline logger."""
    logger = logging.getLogger("veritas")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def timer(stage_name: str, logger: logging.Logger | None = None):
    """Context manager that logs elapsed time for a pipeline stage."""
    log = logger or setup_logging()
    log.info("▶ %s …", stage_name)
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    minutes, seconds = divmod(elapsed, 60)
    if minutes >= 1:
        log.info("✓ %s finished in %dm %.1fs", stage_name, int(minutes), seconds)
    else:
        log.info("✓ %s finished in %.2fs", stage_name, elapsed)
