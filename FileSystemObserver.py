import asyncio
import logging
import os
from asyncio import AbstractEventLoop
from asyncio import Queue

from watchdog.events import FileSystemEventHandler

from LoggerHelper import LoggerHelper


class FileSystemObserver(FileSystemEventHandler):
    """
    A class to observe file system events and handle file creation events.

    This class extends the `FileSystemEventHandler` from `watchdog` to monitor
    file system changes. It listens for file creation events, logs the events,
    and adds the created file names to a processing queue asynchronously.

    Attributes:
        loop (AbstractEventLoop): The event loop used for scheduling tasks.
        queue (Queue): The queue to which filenames are added for processing.
        logger (Logger): The logger used for logging file system events.

    Methods:
        on_any_event(event): Logs any event that occurs on the file system.
        on_created(event): Handles file creation events, logs the event, and
                           adds the created file to the processing queue.
        add_to_queue(filename): Asynchronously adds the filename to the processing queue.
    """

    def __init__(
        self, loop: AbstractEventLoop, queue: Queue, log_level: int = logging.WARNING
    ):
        """
        Initializes the FileSystemObserver.

        Args:
            loop (AbstractEventLoop): The event loop used for scheduling asynchronous tasks.
            queue (Queue): The queue to add filenames to for further processing.
            log_level (int, optional): The logging level for the logger (default is logging.WARNING).
        """
        self.loop = loop
        self.queue = queue
        self.logger = LoggerHelper(
            name="app_logger",
            log_level=log_level,
        ).get_logger()

    def on_any_event(self, event):
        """
        Logs any file system event.

        Args:
            event (FileSystemEvent): The event that occurred on the file system.
        """
        self.logger.debug(event)

    def on_created(self, event):
        """
        Handles file creation events.

        When a file is created, logs the event, extracts the filename, and attempts
        to add the filename to the processing queue asynchronously.

        Args:
            event (FileSystemEvent): The event triggered by file creation.
        """
        # Extract filename from event
        filename = str(event.src_path.split(os.sep)[-1])
        self.logger.info(f"Detected file creation: {filename}")

        # Try to add the filename to the queue
        try:
            asyncio.run_coroutine_threadsafe(self.add_to_queue(filename), self.loop)
        except Exception as e:
            self.logger.warning(f"Exception when adding file to queue: {e}")

    async def add_to_queue(self, filename: str):
        """
        Asynchronously adds the filename to the processing queue.

        Logs the success or failure of adding the filename to the queue.

        Args:
            filename (str): The name of the file to be added to the queue.
        """
        self.logger.info(f"Adding to queue: {filename}")
        await self.queue.put(filename)
        self.logger.info(f"Successfully added: {filename}")
