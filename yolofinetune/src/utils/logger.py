"""
ARGUS-N Logger
Centralised logging for the entire pipeline.
One logger per module, all writing to the same file.
"""

import logging
from pathlib import Path


def get_logger(
    name: str,
    log_path: str = "logs/yolofinetune.log",
    level: str = "INFO"
) -> logging.Logger:

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, level.upper(), logging.INFO))
        console.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(console)

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)

    return logger
