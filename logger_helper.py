import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler


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

    def __init__(
        self,
        name: str,
        log_file: str = "logs/app.log",
        log_level: int = logging.WARNING,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%H:%M:%S",
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler with daily rotation
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = TimedRotatingFileHandler(
                log_file,
                when="midnight",
                interval=1,
                backupCount=7,
                encoding="utf-8",
                utc=False,
            )
            file_handler.setFormatter(formatter)
            file_handler.namer = _custom_namer
            self.logger.addHandler(file_handler)

    def get_logger(self):
        """Returns the configured logger."""
        return self.logger


def _custom_namer(default_name: str) -> str:
    """
    Custom namer function for TimedRotatingFileHandler.
    Renames rotated log files to use the format: app_log_YYYY-MM-DD.log

    Parameters:
        default_name (str): The default filename provided by the handler.

    Returns:
        str: A modified filename including a timestamp in the desired format.
    """
    base, ext = os.path.splitext(default_name)
    base = base.replace(".log", "_log")
    timestamp = time.strftime("%Y-%m-%d")
    return f"{base}_{timestamp}{ext}"
