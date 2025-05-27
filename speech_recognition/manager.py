import asyncio
import threading
from threading import Thread
from typing import Self

from speech_recognition import (
    LoggerHelper,
)
from speech_recognition.workers.abstract_worker import AbstractWorker

log = LoggerHelper(__name__).get_logger()


class Manager:
    """Class to handle requests"""

    __tasks = []

    def __init__(
        self, event_loop, workers: list[AbstractWorker], file_observer
    ) -> None:
        self.__loop = event_loop
        self.__workers = workers
        self.__file_observer = file_observer
        self.__observer_thread = self.__start_file_observer()

    async def start(self) -> Self:
        log.info("Manager starting")
        # Start the observer thread
        self.__observer_thread.start()

        # Create Tasks
        for worker in self.__workers:
            self.__tasks.append(asyncio.create_task(worker.do_work()))

        log.info("Manager started")

        await asyncio.gather(*self.__tasks)

    async def stop(self) -> None:
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
        observer_thread = threading.Thread(
            target=self.__file_observer.start, daemon=True
        )
        return observer_thread
