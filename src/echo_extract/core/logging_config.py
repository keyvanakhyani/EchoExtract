"""Logging configuration for EchoExtract."""

import logging

from rich.logging import RichHandler


def setup_logging(verbose: bool = False) -> None:
    """Configure application-wide logging with rich formatting.

    Args:
        verbose: If True, show DEBUG messages; otherwise INFO and above.
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(name)s | %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    
    # Silence noisy third-party loggers.
    for noisy in ("httpx", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)