import asyncio
import json
import os
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

    __tasks = []

    def __init__(self) -> None:
        # Get configured filepaths, encode, and decode to handle windows paths with \
        self.__IN_DIR = Path(
            config.AUDIO_IN_DIR.encode("unicode_escape").decode()
        ).resolve()
        self.__OUT_DIR = Path(
            config.AUDIO_OUT_DIR.encode("unicode_escape").decode()
        ).resolve()

        # Eventloop
        self.__loop = asyncio.get_running_loop()

        # Queues
        self.__text_queue = asyncio.Queue()
        self.__speech_queue = asyncio.Queue()
        self.__llm_queue = asyncio.Queue()

        # Events
        self.__speech_event = asyncio.Event()
        self.__llm_event = asyncio.Event()

        # Services
        self.__client = WebSocketClient(config.WEBSOCKET_URI, self.__text_queue)
        self.__asr = ASRService()
        self.__llm = LLMService()
        self.__tts = TTSService(self.__text_queue, self.__client)
        self.__file_observer, self.__observer_thread = self.__start_file_observer()

    async def start(self) -> Self:
        log.info("Manager starting")
        # Start the observer thread
        self.__observer_thread.start()

        # register at server as speech recognition service
        await self.__client.connect("sp")

        # Create Tasks
        self.__tasks.append(asyncio.create_task(self.__tts.text_to_speech()))
        self.__tasks.append(asyncio.create_task(self.__handle_audio()))

        log.info("Manager started")

        return self

    async def stop(self) -> None:
        log.info("Shutdown requested.")

        # Disconnect WebSocket client
        log.debug("Closing WebSocket connection")
        await self.__client.close_connection("sp Closing connection")

        # Stop the file observer
        log.debug("Stopping file observer")
        self.__file_observer.stop()

        # Stop all tasks
        log.debug("Stopping all tasks")
        for task in self.__tasks:
            task.cancel()
        await asyncio.gather(*self.__tasks, return_exceptions=True)

        log.info("Shutdown complete.")

    async def ready_for_next_job(self) -> bool:
        # Wait for both services to be done
        log.debug("Waiting for both speech and LLM events to complete...")
        await asyncio.gather(self.__speech_event.wait(), self.__llm_event.wait())

        # Reset both events
        self.__speech_event.clear()
        self.__llm_event.clear()

        log.info("Ready for next job")

        return True

    async def __handle_audio(self) -> None:
        """Convert incoming audio files into text."""

        while True:
            request = await self.__speech_queue.get()
            log.info(f"Received request: {request}")
            await self.__transcribe_audio(request)
            self.__speech_event.set()
            transcript = await self.__llm_queue.get()
            await self.__extract_data_from_transcript(transcript)
            self.__llm_event.set()
            log.info(f"Finished handling request: {request}")

    async def __transcribe_audio(self, request: dict) -> None:
        """Handle a single audio file transcription."""
        filename = request["filename"]
        req_type = request["req_type"]

        if req_type == "BAD_REQUEST":
            log.exception(f"Bad request: {filename}")
            await self.__client.send_message(
                json.dumps(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                        "message": {"text": f"Bad request for file {filename}"},
                    }
                )
            )
            return

        try:
            await self.__client.send_message(
                json.dumps(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_STARTING",
                        "message": {
                            "text": f"Starting Data extraction for file {filename}"
                        },
                    }
                )
            )
            text = self.__asr.transcribe(
                f"{str(self.__IN_DIR)}{os.sep}{filename}",
                f"{str(self.__OUT_DIR)}{os.sep}{filename.rsplit('.', 1)[0]}.wav",
            )
            await self.__llm_queue.put({"prompt": text, "req_type": req_type})
        except Exception as e:
            log.exception(f"Transcription error for {filename}: {e}")
            await self.__client.send_message(
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

    async def __extract_data_from_transcript(self, request: dict) -> None:
        """Handle a single LLM generation request."""
        prompt = request["prompt"]
        req_type = request["req_type"]

        try:
            log.info(f"Prompt received: {prompt} for {req_type}")
            response = self.__llm.generate_json_response(prompt, req_type)
            data = json.loads(response)
            await self.__client.send_message(
                json.dumps(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_SUCCESS",
                        "message": {"text": data},
                    }
                )
            )
        except Exception as e:
            log.exception(f"LLM error: {e}")
            await self.__client.send_message(
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

    def __start_file_observer(self) -> tuple[FileObserver, Thread]:
        observer = FileObserver(self.__loop, self.__speech_queue, str(self.__IN_DIR))
        observer_thread = threading.Thread(target=observer.start, daemon=True)
        return observer, observer_thread
