import asyncio
import json
import threading
from pathlib import Path
from threading import Thread
from typing import Self

from speech_recognition import (
    WebSocketClient,
    config,
    ASRService,
    LLMService,
    LoggerHelper,
    FileObserver,
)
from speech_recognition.services.tts_service import TTSService

log = LoggerHelper(__name__).get_logger()


class Manager:
    """Class to handle requests"""

    _tasks = []
    # Get configured filepaths, encode, and decode to handle windows paths with \
    _IN_DIR = Path(config.AUDIO_IN_DIR.encode("unicode_escape").decode()).resolve()

    _OUT_DIR = Path(config.AUDIO_OUT_DIR.encode("unicode_escape").decode()).resolve()

    def __init__(self) -> None:
        # Eventloop
        self._loop = asyncio.get_running_loop()

        # Queues
        self._text_queue = asyncio.Queue()
        self._speech_queue = asyncio.Queue()
        self._llm_queue = asyncio.Queue()

        # Events
        self._speech_event = asyncio.Event()
        self._llm_event = asyncio.Event()

        # Services
        self._client = WebSocketClient(config.WEBSOCKET_URI, self._text_queue)
        self._asr = ASRService()
        self._llm = LLMService()
        self._tts = TTSService(self._text_queue)
        self._file_observer, self._observer_thread = self._start_file_observer()

    async def start(self) -> Self:
        log.info("Manager starting")
        # Start the observer thread
        self._observer_thread.start()

        # register at server as speech recognition service
        await self._client.connect()
        await self._client.send_message("sp")

        # Create Tasks
        self._tasks.append(asyncio.create_task(self._tts.text_to_speech()))
        self._tasks.append(asyncio.create_task(self._handle_audio()))

        log.info("Manager started")

        return self

    async def ready_for_next_job(self) -> bool:
        # Wait for both services to be done
        log.debug("Waiting for both speech and LLM events to complete...")
        await asyncio.gather(self._speech_event.wait(), self._llm_event.wait())

        # Reset both events
        self._speech_event.clear()
        self._llm_event.clear()

        log.info("Ready for next job")

        return True

    async def stop(self) -> None:
        log.info("Shutdown requested.")

        # Disconnect WebSocket client
        log.debug("Closing WebSocket connection")
        await self._client.close_connection("sp Closing connection")

        # Stop the file observer
        log.debug("Stopping file observer")
        self._file_observer.stop_observer()

        # Stop all tasks
        log.debug("Stopping all tasks")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        log.info("Shutdown complete.")

    async def _handle_audio(
        self,
    ) -> None:
        """Convert incoming audio files into text."""

        while True:
            request = await self._speech_queue.get()
            log.info(f"Received request: {request}")
            await self._transcribe_audio(request)
            self._speech_event.set()
            transcript = await self._llm_queue.get()
            await self._extract_data_from_transcript(transcript)
            self._llm_event.set()
            log.info(f"Finished handling request: {request}")

    async def _transcribe_audio(
        self,
        request: dict,
    ) -> None:
        """Handle a single audio file transcription."""
        filename = request["filename"]
        req_type = request["req_type"]

        if req_type == "BAD_REQUEST":
            log.exception(f"Bad request: {filename}")
            await self._client.send_message(
                json.dumps(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                        "message": {"text": f"Bad request for file {filename}"},
                    }
                )
            )
            return

        try:
            await self._client.send_message(
                json.dumps(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_STARTING",
                        "message": {
                            "text": f"Starting Data extraction for file {filename}"
                        },
                    }
                )
            )
            text = self._asr.transcribe(
                f"{str(self._IN_DIR)}/{filename}",
                f"{str(self._OUT_DIR)}/{filename.rsplit('.', 1)[0]}.wav",
            )
            await self._llm_queue.put({"prompt": text, "req_type": req_type})
        except Exception as e:
            log.exception(f"Transcription error for {filename}: {e}")
            await self._client.send_message(
                json.dumps(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                        "message": {
                            "text": f"Error transcribing file {filename}",
                            "Exception": e,
                        },
                    }
                )
            )

    async def _extract_data_from_transcript(self, request: dict) -> None:
        """Handle a single LLM generation request."""
        prompt = request["prompt"]
        req_type = request["req_type"]

        try:
            log.info(f"Prompt received: {prompt} for {req_type}")
            response = self._llm.generate_json_response(prompt, req_type)
            data = json.loads(response)
            await self._client.send_message(
                json.dumps(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_SUCCESS",
                        "message": {"text": data},
                    }
                )
            )
        except Exception as e:
            log.exception(f"LLM error: {e}")
            await self._client.send_message(
                json.dumps(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                        "message": {
                            "text": f"Error while processing prompt {prompt} for request {req_type}",
                            "exception": e,
                        },
                    }
                )
            )

    def _start_file_observer(self) -> tuple[FileObserver, Thread]:
        observer = FileObserver(self._loop, self._speech_queue)
        observer_thread = threading.Thread(
            target=observer.start_observer, args=(self._IN_DIR,), daemon=True
        )
        return observer, observer_thread
