import asyncio
import os
from asyncio import AbstractEventLoop
from asyncio import Queue
from typing import Any

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

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
        __loop (AbstractEventLoop): The event loop used for scheduling tasks.
        __queue (Queue): The queue to which filenames are added for processing.
    """

    def __init__(self, loop: AbstractEventLoop, queue: Queue, path: str) -> None:
        self.__loop = loop
        self.__queue = queue
        self.__path = path

        # Used to store the actual observer instance
        self.__observer = None

    def start(self) -> None:
        """
        Starts the file system observer to monitor the specified directory.
        """
        self.__observer = Observer()
        log.debug(f"Starting file observer on {self.__path}")
        self.__observer.schedule(event_handler=self, path=self.__path, recursive=False)
        self.__observer.start()
        log.info(f"[Observer] Started watching {self.__path}")
        self.__observer.join()

    def stop(self) -> None:
        """
        Gracefully stops the file system observer.
        This method stops the observer and joins the threads to ensure a clean shutdown.
        """
        if self.__observer is not None:
            self.__observer.stop()
            log.info("[Observer] Stopping observer...")
            # Wait for the observer thread to finish
            self.__observer.join()
            log.info("[Observer] Observer stopped successfully.")
        else:
            log.warning("[Observer] No observer is currently running.")

    def on_created(self, event: FileSystemEvent) -> None:
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
                self.__add_to_queue({"file": event.src_path, "req_type": req_type}),
                self.__loop,
            )
        except Exception as e:
            log.warning(f"Exception when adding file to queue: {e}")

    async def __add_to_queue(self, item: Any) -> None:
        """
        Asynchronously adds an item to the processing queue.

        Args:
            item (Any): The item to be added to the queue.
        """
        log.debug(f"Adding to queue: {item}")
        await self.__queue.put(item)
        log.debug(f"Successfully added: {item}")
