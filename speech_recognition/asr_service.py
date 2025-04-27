import os
import time

import pydub
import torch
from pydub.silence import detect_nonsilent
from transformers import pipeline, Pipeline

from speech_recognition import config
from speech_recognition.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


def is_audio_empty(
    infile: str, min_silence_len: int = 100, silence_thresh: int = -50
) -> bool:
    """Check if an audio file is empty or contains only silence.

    Args:
        infile (str): Path to the input audio file.
        min_silence_len (int, optional): Minimum length of silence in milliseconds to consider. Defaults to 100.
        silence_thresh (int, optional): Silence threshold in dBFS. Defaults to -50.

    Returns:
        bool: True if the audio is silent, False otherwise.
    """
    audio = pydub.AudioSegment.from_file(infile)

    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    return len(nonsilent) == 0


def is_file_empty(infile: str) -> bool:
    """Check if a file is empty based on file size or audio content.

    Args:
        infile (str): Path to the input file.

    Returns:
        bool: True if the file is empty or contains only silence, False otherwise.
    """
    size_kb = os.path.getsize(infile) / 1024
    if size_kb <= 4:
        return True
    return is_audio_empty(infile)


def convert_audio_to_wav(infile: str, outfile: str) -> None:
    """Convert an input audio file to WAV format.

    Args:
        infile (str): Path to the input audio file.
        outfile (str): Path to save the output WAV file.

    Returns:
        None
    """
    log.info(f"Converting {infile} to WAV format as {outfile}")
    try:
        sound = pydub.AudioSegment.from_file(infile)
        sound.export(outfile, format="wav")
    except Exception as e:
        log.exception(f"Error during conversion of {infile} to WAV format: {e}")
        raise TranscriptionError(f"Error during conversion of {infile} to WAV format")


class TranscriptionError(Exception):
    """Exception raised for transcription errors.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message):
        """Initialize TranscriptionError.

        Args:
            message (str): Error message to describe the exception.
        """
        self.message = message
        super().__init__(self.message)


class ASRService:
    """Automatic Speech Recognition (ASR) service using a Hugging Face pipeline."""

    def __init__(self) -> None:
        """Initialize the ASR service with configuration from config.py."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = config.ASR_LANGUAGE
        self.model_name = config.ASR_MODEL_NAME
        self.transcriber = self._load_model()

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
        if is_file_empty(infile):
            raise TranscriptionError(f"file {infile} is empty or contains only silence")

        convert_audio_to_wav(infile, outfile)
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
