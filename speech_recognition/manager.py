import asyncio
import threading
from asyncio import AbstractEventLoop
from threading import Thread

from speech_recognition import (
    LoggerHelper,
    FileObserver,
)
from speech_recognition.workers.abstract_worker import AbstractWorker

log = LoggerHelper(__name__).get_logger()


class Manager:
    """
    Orchestrates workers and a file observer within an asyncio event loop.

    This class manages the lifecycle of multiple asynchronous workers and a file observer
    running in a separate thread. It starts the file observer in a daemon thread and runs
    all workers as asyncio tasks on the provided event loop. The manager handles clean
    startup and shutdown procedures for all components.

    Attributes:
        __tasks (list[asyncio.Task]): List of asyncio tasks corresponding to worker coroutines.
        __loop (AbstractEventLoop): The asyncio event loop on which workers run.
        __workers (list[AbstractWorker]): The list of workers to manage.
        __file_observer (FileObserver): File system observer to watch for file events.
        __observer_thread (Thread): Thread running the file observer.
    """

    __tasks = []

    def __init__(
        self,
        loop: AbstractEventLoop,
        workers: list[AbstractWorker],
        file_observer: FileObserver,
    ) -> None:
        """
        Initialize the Manager with an event loop, workers, and a file observer.

        Args:
            loop (AbstractEventLoop): The asyncio event loop to schedule worker tasks.
            workers (list[AbstractWorker]): List of worker instances implementing do_work().
            file_observer (FileObserver): The file observer to monitor file system changes.
        """
        self.__loop = loop
        self.__workers = workers
        self.__file_observer = file_observer
        self.__observer_thread = self.__start_file_observer()

    async def start(self) -> None:
        """
        Start the manager, launching the file observer thread and worker tasks.

        This method:
        - Starts the file observer in a separate daemon thread.
        - Creates asyncio tasks for each worker’s `do_work()` coroutine.
        - Awaits all worker tasks concurrently.
        """
        log.info("Manager starting")
        # Start the observer thread
        self.__observer_thread.start()

        # Create Tasks
        for worker in self.__workers:
            self.__tasks.append(asyncio.create_task(worker.do_work()))

        log.info("Manager started")

        await asyncio.gather(*self.__tasks)

    async def stop(self) -> None:
        """
        Stop the manager, shutting down the file observer and cancelling worker tasks.

        This method:
        - Stops the file observer.
        - Cancels all running worker tasks.
        - Awaits task cancellation, suppressing exceptions.
        - Logs shutdown progress and completion.
        """
        log.info("Stopping manager.")

        # Stop the file observer
        log.debug("Stopping file observer")
        self.__file_observer.stop()

        # Stop all tasks
        log.debug("Stopping all tasks")
        for task in self.__tasks:
            task.cancel()
        await asyncio.gather(*self.__tasks, return_exceptions=True)

        log.info("Manager stopped.")

    def __start_file_observer(self) -> Thread:
        """
        Create and return a daemon thread to run the file observer.

        Returns:
            Thread: A daemon thread running the file observer's start method.
        """
        observer_thread = threading.Thread(
            target=self.__file_observer.start, daemon=True
        )
        return observer_thread
