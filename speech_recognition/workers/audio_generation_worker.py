import asyncio
import os

from speech_recognition import WebSocketClient
from speech_recognition.exceptions.audio_generation_error import AudioGenerationError
from speech_recognition.services.tts_service import TTSService
from speech_recognition.workers.abstract_worker import AbstractWorker


class AudioGenerationWorker(AbstractWorker):
    """Worker that listens for text requests, generates audio, and reports back via WebSocket.

    Attributes:
        __text_queue (asyncio.Queue): Queue of text strings to convert to audio.
        __tts_service (TTSService): Service to generate audio from text.
        __client (WebSocketClient): WebSocket client for sending responses.
    """

    def __init__(
        self,
        text_queue: asyncio.Queue,
        tts_service: TTSService,
        client: WebSocketClient,
    ) -> None:
        """Initialize the AudioGenerationWorker.

        Args:
            text_queue (asyncio.Queue): Queue where text requests are received.
            tts_service (TTSService): Instance to perform text-to-speech generation.
            client (WebSocketClient): WebSocket client to send status and results.
        """
        self.__text_queue = text_queue
        self.__tts_service = tts_service
        self.__client = client

    async def do_work(self) -> None:
        """Continuously process text-to-audio generation requests from the queue.

        Waits for new text input from the queue, generates the audio file,
        and sends a success or error message back through the WebSocket client.
        """
        while True:
            request = await self.__text_queue.get()
            try:
                res = str(await self.__tts_service.generate_audio(request))
                filename = res.split(os.sep)[-1].rstrip()
                await self.__client.send_message(
                    {
                        "type": "GENERATE_AUDIO_SUCCESS",
                        "message": {
                            "text": f"Successfully generated audio file: {filename}",
                        },
                    }
                )
            except AudioGenerationError as e:
                await self.__client.send_message(
                    {
                        "type": "GENERATE_AUDIO_ERROR",
                        "message": {"text": f"{e.message}"},
                    }
                )
