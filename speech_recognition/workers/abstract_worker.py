from abc import ABC, abstractmethod


class AbstractWorker(ABC):
    """Abstract base class for worker implementations.

    Subclasses must implement the async `do_work` method to define
    the worker's main loop or task execution logic.
    """

    @abstractmethod
    async def do_work(self):
        """Start the worker loop.

        This method should be implemented by subclasses to define
        the async work to be performed, often running an infinite
        loop processing jobs or events.
        """
        pass
