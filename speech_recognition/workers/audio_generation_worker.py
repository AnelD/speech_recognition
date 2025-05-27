import os

from speech_recognition.exceptions.audio_generation_error import AudioGenerationError
from speech_recognition.workers.abstract_worker import AbstractWorker


class AudioGenerationWorker(AbstractWorker):
    def __init__(self, text_queue, tts_service, client):
        self.text_queue = text_queue
        self.tts_service = tts_service
        self.client = client

    async def do_work(self):
        while True:
            request = await self.text_queue.get()
            try:
                res = str(await self.tts_service.generate_audio(request))
                filename = res.split(os.sep)[-1].rstrip()
                await self.client.send_message(
                    {
                        "type": "GENERATE_AUDIO_SUCCESS",
                        "message": {
                            "text": f"Successfully generated audio file: {filename}",
                        },
                    }
                )
            except AudioGenerationError as e:
                await self.client.send_message(
                    {
                        "type": "GENERATE_AUDIO_ERROR",
                        "message": {"text": f"{e.message}"},
                    }
                )
