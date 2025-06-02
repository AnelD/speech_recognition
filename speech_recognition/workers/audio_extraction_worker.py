import asyncio
import os

from speech_recognition import LoggerHelper, ASRService, LLMService, WebSocketClient
from speech_recognition.exceptions.llm_processing_error import LLMProcessingError
from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.workers.abstract_worker import AbstractWorker

log = LoggerHelper(__name__).get_logger()


class AudioExtractionWorker(AbstractWorker):
    def __init__(
        self,
        speech_queue: asyncio.Queue,
        asr_service: ASRService,
        llm_service: LLMService,
        client: WebSocketClient,
    ):
        self.__speech_queue = speech_queue
        self.__asr_service = asr_service
        self.__llm_service = llm_service
        self.__client = client

    async def do_work(self):
        while True:
            request = await self.__speech_queue.get()
            log.info(f"Received request: {request}")
            file = request["file"]
            req_type = request["req_type"]

            if req_type == "BAD_REQUEST":
                log.error(f"Bad request: {req_type}")
                await self.__client.send_message(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                        "message": {"text": f"Bad request for file {file}"},
                    }
                )
                continue

            try:
                await self.__client.send_message(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_STARTING",
                        "message": {
                            "text": f"Starting Data extraction for file: {file.split(os.sep)[-1]}",
                        },
                    }
                )
                text = await asyncio.to_thread(self.__asr_service.transcribe, file)
                result = await asyncio.to_thread(
                    self.__llm_service.generate_json_response, text, request["req_type"]
                )
                await self.__client.send_message(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_SUCCESS",
                        "message": {"text": result},
                    }
                )

            except (LLMProcessingError, TranscriptionError) as e:
                log.exception(
                    f"Error while extracting data from: {file.split(os.sep)[-1]}: {e}"
                )
                await self.__client.send_message(
                    {
                        "type": "EXTRACT_DATA_FROM_AUDIO_ERROR",
                        "message": {"text": e.message},
                    }
                )
