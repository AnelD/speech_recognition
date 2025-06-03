import logging
import os
from logging.handlers import TimedRotatingFileHandler

import colorlog

from speech_recognition import config


class LoggerHelper:
    """Helper class to configure and retrieve a logger with console and file output.

    Features:
        - Console logging with color output using `colorlog`.
        - File logging with daily rotation using `TimedRotatingFileHandler`.
        - Keeps logs for up to 7 days.

    Attributes:
        logger (logging.Logger): The configured logger instance.
        log_file (str): The log file path defined in `config.LOG_FILE`.
    """

    def __init__(self, name: str) -> None:
        """Initializes the LoggerHelper instance.

        Sets up the logger with colored console output and daily-rotating log files
        unless in a test environment.

        Args:
            name (str): Name for the logger, typically `__name__`.
        """
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
                delay=True,
            )
            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """Returns the configured logger instance.

        Returns:
            logging.Logger: The configured logger.
        """
        return self.logger
