import time

import torch
from transformers import pipeline, Pipeline

from speech_recognition import config
from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.utils.audio_helper import AudioHelper
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class ASRService:
    """Automatic Speech Recognition (ASR) service using a Hugging Face pipeline."""

    def __init__(self) -> None:
        """Initialize the ASR service with configuration from config.py."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = config.ASR_LANGUAGE
        self.model_name = config.ASR_MODEL_NAME
        self.audio_helper = AudioHelper()
        self.transcriber = self._load_model()

    def transcribe(self, infile: str, outfile: str) -> str:
        """Transcribe an audio file to text.

        Args:
            infile (str): Path to the input audio file.
            outfile (str): Path to save the intermediate WAV file.

        Returns:
            str: Transcribed text.

        Raises:
            TranscriptionError: If the file is empty or an error occurs during transcription.
        """
        if self.audio_helper.is_file_empty(infile):
            raise TranscriptionError(f"file {infile} is empty or contains only silence")

        self.audio_helper.convert_audio_to_wav(infile, outfile)
        log.info(f"Transcribing {outfile}...")
        t0 = time.time()

        try:
            result = self.transcriber(
                outfile, generate_kwargs={"language": self.language}
            )
        except Exception as e:
            log.exception(f"Error while transcribing: {e}")
            raise TranscriptionError(f"Error while transcribing file: {outfile}")

        t1 = time.time()
        log.info(f"Transcription completed in {t1 - t0:.2f} seconds.")
        return result["text"]

    def _load_model(self) -> Pipeline:
        """Load the ASR model.

        Returns:
            Pipeline: Loaded Hugging Face pipeline for transcription.
        """
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": (
                # if left on auto sets float16 for cpu which results in very slow transcriptions
                torch.float16
                if self.device.type == "cuda"
                else torch.float32
            ),
        }
        log.info(
            f"Loading Whisper: {self.model_name} with model kwargs: {model_kwargs} on device: {self.device}"
        )
        t0 = time.time()
        model = pipeline(
            task="automatic-speech-recognition",
            model=self.model_name,
            # Makes chunks of audio with length x
            chunk_length_s=30,
            model_kwargs=model_kwargs,
            # Explicitly loading the model onto a device also slows down both on cpu and gpu ???
        )
        t1 = time.time()
        log.info(f"Whisper model loaded in {t1 - t0:.2f} seconds.")
        return model
