import asyncio
import logging
import os
from asyncio import AbstractEventLoop
from asyncio import Queue

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from LoggerHelper import LoggerHelper


class FileSystemObserver(FileSystemEventHandler):
    """
    A class to observe file system events and handle file creation events.
    """

    def __init__(
        self, loop: AbstractEventLoop, queue: Queue, log_level: int = logging.WARNING
    ):
        self.loop = loop
        self.queue = queue
        self.logger = LoggerHelper(
            name="app_logger",
            log_level=log_level,
        ).get_logger()

        # Used to store the actual observer instance
        self.observer = None

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

        Args:
            event (FileSystemEvent): The event triggered by file creation.
        """
        filename = str(event.src_path.split(os.sep)[-1])
        self.logger.info(f"Detected file creation: {filename}")

        try:
            asyncio.run_coroutine_threadsafe(self.add_to_queue(filename), self.loop)
        except Exception as e:
            self.logger.warning(f"Exception when adding file to queue: {e}")

    async def add_to_queue(self, filename: str):
        """
        Asynchronously adds the filename to the processing queue.

        Args:
            filename (str): The name of the file to be added to the queue.
        """
        self.logger.info(f"Adding to queue: {filename}")
        await self.queue.put(filename)
        self.logger.info(f"Successfully added: {filename}")

    def start_observer(self, path: str):
        """
        Starts the file system observer to monitor the specified directory.

        Args:
            path (str): The directory path to monitor for file system events.
        """
        event_handler = self  # Use the current instance as the event handler
        self.observer = Observer()
        self.observer.schedule(event_handler, path, recursive=False)
        self.observer.start()
        self.logger.info(f"[Observer] Started watching {path}")
        self.observer.join()

    def stop_observer(self):
        """
        Gracefully stops the file system observer.
        This method stops the observer and joins the threads to ensure a clean shutdown.
        """
        if self.observer is not None:
            self.observer.stop()
            self.logger.info("[Observer] Stopping observer...")
            # Wait for the observer thread to finish
            self.observer.join()
            self.logger.info("[Observer] Observer stopped successfully.")
        else:
            self.logger.warning("[Observer] No observer is currently running.")
