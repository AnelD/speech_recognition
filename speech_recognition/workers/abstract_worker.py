from abc import ABC, abstractmethod


class AbstractWorker(ABC):
    @abstractmethod
    async def do_work(self):
        """Start the worker loop."""
        pass
