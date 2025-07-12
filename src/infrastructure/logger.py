import logging
import os
import sys
from logging import Logger

from src.infrastructure.config import ENV
from src.infrastructure.paths import PROJECT_ROOT


def get_logger(name: str) -> Logger:
    log_handlers = [logging.StreamHandler(sys.stdout)]

    if ENV != "production":
        os.makedirs("../../logs", exist_ok=True)
        log_handlers.append(logging.FileHandler(PROJECT_ROOT / "logs" / f"{name}.log"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=log_handlers,
    )
    logger = logging.getLogger(name)
    return logger
