import logging
import sys

FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_console_handler() -> logging.StreamHandler:  # type: ignore
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.addHandler(get_console_handler())
    logger.propagate = True
    return logger
