import asyncio
import os

from speech_recognition import WebSocketClient
from speech_recognition.exceptions.audio_generation_error import AudioGenerationError
from speech_recognition.services.tts_service import TTSService
from speech_recognition.workers.abstract_worker import AbstractWorker


class AudioGenerationWorker(AbstractWorker):
    def __init__(
        self,
        text_queue: asyncio.Queue,
        tts_service: TTSService,
        client: WebSocketClient,
    ) -> None:
        self.__text_queue = text_queue
        self.__tts_service = tts_service
        self.__client = client

    async def do_work(self) -> None:
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
