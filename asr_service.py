import logging
import time

import pydub
import torch
from transformers import pipeline

from logger_helper import LoggerHelper

log = LoggerHelper(__name__, log_level=logging.DEBUG).get_logger()


def convert_audio_to_wav(infile: str, outfile: str):
    """Convert an input audio file to WAV format."""
    log.debug(f"Converting {infile} to WAV format as {outfile}")
    sound = pydub.AudioSegment.from_file(infile)
    sound.export(outfile, format="wav")


class ASRService:
    def __init__(self, model_name="openai/whisper-large-v3-turbo", language="german"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.model_name = model_name
        self.transcriber = self._load_model()

    def _load_model(self):
        log.info(f"Using device: {self.device}")
        t0 = time.time()
        model = pipeline(model=self.model_name, device=self.device)
        t1 = time.time()
        log.info(f"Whisper model loaded in {t1 - t0:.2f} seconds.")
        return model

    def transcribe(self, infile: str, outfile: str):
        convert_audio_to_wav(infile, outfile)
        log.info(f"Transcribing {outfile}...")
        t0 = time.time()
        result = self.transcriber(outfile, generate_kwargs={"language": self.language})
        t1 = time.time()
        log.info(f"Transcription completed in {t1 - t0:.2f} seconds.")
        return result["text"]
