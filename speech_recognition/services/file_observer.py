import asyncio
import os
from asyncio import AbstractEventLoop
from typing import Any

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from speech_recognition.services.llm_service import RequestType
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class FileObserver(FileSystemEventHandler):
    """Observes a directory and handles file creation events.

    This class uses `watchdog` to monitor a specified directory. When a new file is
    created, it identifies the type of request based on the filename and enqueues
    the file for asynchronous processing.

    Attributes:
        __loop (AbstractEventLoop): The event loop used to schedule coroutines.
        __queue (asyncio.Queue): An asyncio queue to store files for processing.
        __path (str): Directory path to be observed.
        __observer (BaseObserver): The watchdog observer instance.
    """

    def __init__(
        self, loop: AbstractEventLoop, queue: asyncio.Queue, path: str
    ) -> None:
        """Initializes the FileObserver.

        Args:
            loop (AbstractEventLoop): The event loop for running coroutines.
            queue (asyncio.Queue): An asyncio queue where detected files will be added.
            path (str): The path to the directory to observe.
        """
        self.__loop = loop
        self.__queue = queue
        self.__path = path
        self.__observer = None

    def start(self) -> None:
        """Starts monitoring the target directory for file creation events."""
        self.__observer = Observer()
        log.debug(f"Starting file observer on {self.__path}")
        self.__observer.schedule(event_handler=self, path=self.__path, recursive=False)
        self.__observer.start()
        log.info(f"[Observer] Started watching {self.__path}")
        self.__observer.join()

    def stop(self) -> None:
        """Stops the file observer and waits for its thread to terminate."""
        if self.__observer is not None:
            self.__observer.stop()
            log.info("[Observer] Stopping observer...")
            self.__observer.join()
            log.info("[Observer] Observer stopped successfully.")
        else:
            log.warning("[Observer] No observer is currently running.")

    def on_created(self, event: FileSystemEvent) -> None:
        """Handles the event triggered when a new file is created.

        Parses the file name to determine the request type and adds the file path
        and request type to the queue for processing.

        Args:
            event (FileSystemEvent): The file creation event.
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
        """Adds an item to the asyncio queue.

        Args:
            item (Any): The item to add to the queue, typically a dict containing
                'file' (str): File path, and 'req_type' (RequestType).
        """
        log.debug(f"Adding to queue: {item}")
        await self.__queue.put(item)
        log.debug(f"Successfully added: {item}")
