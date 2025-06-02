import logging
import os
from logging.handlers import TimedRotatingFileHandler

import colorlog

from speech_recognition import config


class LoggerHelper:
    """
    LoggerHelper sets up a logger that writes to both the console and a time-rotated log file.

    Logs are:
    - Written to the console with timestamps (hour:min:sec)
    - Rotated daily at midnight
    - Stored for up to 7 days
    - Saved with filenames formatted like: logs/app_log_YYYY-MM-DD.log

    Attributes:
        name (str): Name of the logger.
        log_file (str): Path to the log file. Default is 'logs/app.log'.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """

    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(config.LOG_LEVEL)
        self.logger.propagate = False
        self.log_file = config.LOG_FILE

        if not self.logger.handlers:
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # Omit the file handler when running tests
            if os.getenv("TEST"):
                return

            # File handler with daily rotation
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            file_handler = TimedRotatingFileHandler(
                self.log_file,
                when="midnight",
                interval=1,
                backupCount=7,
                encoding="utf-8",
                utc=False,
            )
            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Returns the configured logger."""
        return self.logger
