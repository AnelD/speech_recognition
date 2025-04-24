import logging
import time

import pydub
import torch
from transformers import pipeline

from LoggerHelper import LoggerHelper

logger = LoggerHelper("app_log", log_level=logging.DEBUG).get_logger()


class ASRService:
    def __init__(self, model_name="openai/whisper-large-v3-turbo", language="german"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.model_name = model_name
        self.transcriber = self._load_model()

    def _load_model(self):
        logger.info(f"Using device: {self.device}")
        t0 = time.time()
        model = pipeline(model=self.model_name, torch_dtype="auto", device=self.device)
        t1 = time.time()
        logger.info(f"Whisper model loaded in {t1 - t0:.2f} seconds.")
        return model

    def convert_audio_to_wav(self, infile: str, outfile: str):
        """Convert input audio file to WAV format."""
        logger.debug(f"Converting {infile} to WAV format as {outfile}")
        sound = pydub.AudioSegment.from_file(infile)
        sound.export(outfile, format="wav")

    def transcribe(self, infile: str, outfile: str):
        self.convert_audio_to_wav(infile, outfile)
        logger.info(f"Transcribing {outfile}...")
        t0 = time.time()
        result = self.transcriber(outfile, generate_kwargs={"language": self.language})
        t1 = time.time()
        logger.info(f"Transcription completed in {t1 - t0:.2f} seconds.")
        return result["text"]
