import asyncio
import os

from speech_recognition import LoggerHelper, ASRService, LLMService, WebSocketClient
from speech_recognition.exceptions.llm_processing_error import LLMProcessingError
from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.workers.abstract_worker import AbstractWorker

log = LoggerHelper(__name__).get_logger()


class AudioExtractionWorker(AbstractWorker):
    """
    Worker that processes audio files to extract text and structured data.

    This worker listens for incoming audio processing requests on a queue, transcribes
    the audio to text using an ASR (Automatic Speech Recognition) service, and then
    generates structured JSON data using a language model (LLM) service based on the transcription.

    It communicates progress, success, and errors back to a client via WebSocket messages.

    Attributes:
        __speech_queue (asyncio.Queue): Queue of audio processing requests.
        __asr_service (ASRService): Service to transcribe audio to text.
        __llm_service (LLMService): Service to generate structured JSON response from text.
        __client (WebSocketClient): Client to send status and result messages.
    """

    def __init__(
        self,
        speech_queue: asyncio.Queue,
        asr_service: ASRService,
        llm_service: LLMService,
        client: WebSocketClient,
    ):
        """
        Initialize the AudioExtractionWorker.

        Args:
            speech_queue (asyncio.Queue): Queue from which audio processing requests are read.
            asr_service (ASRService): Instance of the ASR service for audio transcription.
            llm_service (LLMService): Instance of the LLM service for JSON response generation.
            client (WebSocketClient): WebSocket client used to send messages back to the requester.
        """
        self.__speech_queue = speech_queue
        self.__asr_service = asr_service
        self.__llm_service = llm_service
        self.__client = client

    async def do_work(self):
        """
        Continuously process audio extraction requests from the queue.

        For each request:
        - Validate the request type.
        - Notify the client that processing is starting.
        - Transcribe the audio file to text using ASR service.
        - Generate a JSON response from the transcription using the LLM service.
        - Send success or error messages back to the client.

        Handles exceptions from transcription and LLM processing and sends error messages accordingly.
        """
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
