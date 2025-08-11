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


# region Prefix


class PrefixAdapter(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        return f"[{self.prefix}]\t{msg}", kwargs


def get_prefixed_logger(name: str, prefix: str | None = None) -> logging.LoggerAdapter:
    logger = get_logger(name)
    prefixed_logger = PrefixAdapter(logger, prefix if prefix is not None else name)
    return prefixed_logger


# endregion

# if __name__ == "__main__":
#     logger = get_logger("test")
#     prefixed_logger = get_prefixed_logger("test", "PREFIX")
#
#     logger.info("Обычное сообщение")
#     prefixed_logger.info("Сообщение с префиксом")
