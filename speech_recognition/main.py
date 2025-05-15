import asyncio

from speech_recognition import (
    LoggerHelper,
)
from speech_recognition.manager import Manager

log = LoggerHelper(__name__).get_logger()


async def main():
    log.info("Initializing...")
    manager = Manager().start()
    log.info("Initialization complete.")
    try:
        while True:
            await manager.ready_for_next_job()

    # Graceful Shutdown,
    # when closed with, for example, CTRL+C the currently running tasks raise CancelledError
    # The actual KeyboardInterrupt is raised outside the event loop
    except asyncio.CancelledError:
        await manager.stop()
        log.info("Cancellation complete.")


if __name__ == "__main__":
    asyncio.run(main())
