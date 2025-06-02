import asyncio
from pathlib import Path

import art

from speech_recognition import (
    LoggerHelper,
    FileObserver,
    config,
    ASRService,
    LLMService,
    WebSocketClient,
)
from speech_recognition.manager import Manager
from speech_recognition.services.tts_service import TTSService
from speech_recognition.workers.audio_extraction_worker import AudioExtractionWorker
from speech_recognition.workers.audio_generation_worker import AudioGenerationWorker

log = LoggerHelper(__name__).get_logger()


async def main():
    log.info("Initializing...")
    # Get configured filepaths, encode, and decode to handle windows paths with \
    in_dir = str(Path(config.AUDIO_IN_DIR.encode("unicode_escape").decode()).resolve())

    # Create Queues
    speech_queue = asyncio.Queue()
    text_queue = asyncio.Queue()

    # Get current eventloop
    event_loop = asyncio.get_running_loop()

    # Start the services
    client = WebSocketClient(config.WEBSOCKET_URI, text_queue)
    asr = ASRService()
    llm = LLMService()
    tts = TTSService(client)
    file_observer = FileObserver(event_loop, speech_queue, in_dir)

    # Create Workers
    stt_worker = AudioExtractionWorker(speech_queue, asr, llm, client)
    tts_worker = AudioGenerationWorker(text_queue, tts, client)

    manager = Manager(event_loop, [stt_worker, tts_worker], file_observer)
    await client.connect("sp")

    log.info("Initialization complete.")
    art.tprint("speech", "sub-zero")
    art.tprint("recognition", "sub-zero")
    art.tprint("started", "sub-zero")
    try:
        await manager.start()

    # Graceful Shutdown,
    # when closed with, for example, CTRL+C the currently running tasks raise CancelledError
    # The actual KeyboardInterrupt is raised outside the event loop
    except asyncio.CancelledError:
        log.info("Cancellation requested.")
        await manager.stop()
        log.info("Cancellation complete.")
        art.tprint("speech", "sub-zero")
        art.tprint("recognition", "sub-zero")
        art.tprint("stopped", "sub-zero")


if __name__ == "__main__":
    asyncio.run(main())
