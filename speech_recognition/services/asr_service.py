import time

import torch
from transformers import pipeline, Pipeline

from speech_recognition import config
from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.utils.audio_helper import AudioHelper
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class ASRService:
    """ "Automatic Speech Recognition (ASR) service using a Hugging Face pipeline.

    This class loads a Whisper model and provides a method to transcribe audio files.

    Attributes:
        __device (torch.device): The device (CPU or CUDA) on which the model runs.
        __language (str): Language used for transcription, from config.
        __model_name (str): Model identifier from Hugging Face used for ASR.
        __audio_helper (AudioHelper): Helper class for audio file manipulation.
        __transcriber (Pipeline): Hugging Face pipeline used for speech recognition.
    """

    def __init__(self) -> None:
        """Initializes the ASRService.

        Sets up the device (CPU/GPU), loads configuration values, initializes audio helper utilities,
        and loads the ASR model pipeline.
        """
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__language = config.ASR_LANGUAGE
        self.__model_name = config.ASR_MODEL_NAME
        self.__audio_helper = AudioHelper()
        self.__transcriber = self.__load_model()

    def transcribe(self, file: str) -> str:
        """Transcribes an audio file to text using the loaded ASR model.

        Args:
            file (str): Path to the input audio file.

        Returns:
            str: The transcribed text from the audio.

        Raises:
            TranscriptionError: If the audio file is empty or an error occurs during transcription.
        """
        if self.__audio_helper.is_file_empty(file):
            raise TranscriptionError(f"file {file} is empty or contains only silence")

        outfile = self.__audio_helper.convert_audio_to_wav(file)
        log.info(f"Transcribing {outfile}...")
        t0 = time.time()

        try:
            result = self.__transcriber(
                outfile, generate_kwargs={"language": self.__language}
            )
        except Exception as e:
            log.exception(f"Error while transcribing: {e}")
            raise TranscriptionError(f"Error while transcribing file: {outfile}")

        t1 = time.time()
        log.info(f"Transcription completed in {t1 - t0:.2f} seconds.")
        return result["text"]

    def __load_model(self) -> Pipeline:
        """Loads the Whisper ASR model using the Hugging Face Transformers pipeline.

        The model is configured based on the available hardware (GPU/CPU) and loaded with
        appropriate settings for chunk size and precision.

        Returns:
            Pipeline: A Hugging Face Transformers pipeline for automatic speech recognition.
        """
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": (
                # If left on 'auto', sets float16 for CPU which results in very slow transcriptions
                torch.float16
                if self.__device.type == "cuda"
                else torch.float32
            ),
        }

        log.info(
            f"Loading Whisper: {self.__model_name} with model kwargs: {model_kwargs} on device: {self.__device}"
        )
        t0 = time.time()
        model = pipeline(
            task="automatic-speech-recognition",
            model=self.__model_name,
            chunk_length_s=30,
            model_kwargs=model_kwargs,
        )
        t1 = time.time()
        log.info(f"Whisper model loaded in {t1 - t0:.2f} seconds.")
        return model
