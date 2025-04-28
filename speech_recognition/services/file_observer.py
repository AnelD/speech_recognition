import asyncio
import os
from asyncio import AbstractEventLoop
from asyncio import Queue
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from speech_recognition import config
from speech_recognition.services.llm_service import RequestType
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class FileObserver(FileSystemEventHandler):
    """
    A class to observe file system events and handle file creation events.

    This class extends the `FileSystemEventHandler` from `watchdog` to monitor
    file system changes. It listens for file creation events, logs the events,
    and adds the created file names to a processing queue asynchronously.

    Attributes:
        loop (AbstractEventLoop): The event loop used for scheduling tasks.
        queue (Queue): The queue to which filenames are added for processing.
    """

    def __init__(self, loop: AbstractEventLoop, queue: Queue) -> None:
        self.loop = loop
        self.queue = queue

        # Used to store the actual observer instance
        self.observer = None

        self.log_all_events = config.LOG_ALL_EVENTS

    def on_any_event(self, event) -> None:
        """
        Logs any file system event.

        Args:
            event (FileSystemEvent): The event that occurred on the file system.
        """
        if self.log_all_events:
            log.debug(event)

    def on_created(self, event) -> None:
        """
        Handles file creation events.

        Args:
            event (FileSystemEvent): The event triggered by file creation.
        """
        filename = str(event.src_path.split(os.sep)[-1])
        log.info(f"Detected file creation: {filename}")
        match filename.split("-")[0]:
            case "person":
                req_type = RequestType.PERSON_DATA
            case "command":
                req_type = RequestType.COMMAND
            case _:
                req_type = RequestType.BAD_REQUEST

        try:
            asyncio.run_coroutine_threadsafe(
                self._add_to_queue({"filename": filename, "req_type": req_type}),
                self.loop,
            )
        except Exception as e:
            log.warning(f"Exception when adding file to queue: {e}")

    async def _add_to_queue(self, item: Any) -> None:
        """
        Asynchronously adds an item to the processing queue.

        Args:
            item (Any): The item to be added to the queue.
        """
        log.debug(f"Adding to queue: {item}")
        await self.queue.put(item)
        log.debug(f"Successfully added: {item}")

    def start_observer(self, path: str) -> None:
        """
        Starts the file system observer to monitor the specified directory.

        Args:
            path (str): The directory path to monitor for file system events.
        """
        event_handler = self  # Use the current instance as the event handler
        self.observer = Observer()
        self.observer.schedule(event_handler, path, recursive=False)
        self.observer.start()
        log.info(f"[Observer] Started watching {path}")
        self.observer.join()

    def stop_observer(self) -> None:
        """
        Gracefully stops the file system observer.
        This method stops the observer and joins the threads to ensure a clean shutdown.
        """
        if self.observer is not None:
            self.observer.stop()
            log.info("[Observer] Stopping observer...")
            # Wait for the observer thread to finish
            self.observer.join()
            log.info("[Observer] Observer stopped successfully.")
        else:
            log.warning("[Observer] No observer is currently running.")
