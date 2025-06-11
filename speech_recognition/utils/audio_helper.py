import os
import subprocess
from pathlib import Path
from typing import Set

import pydub
from pydub.silence import detect_nonsilent

from speech_recognition import config
from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class AudioHelper:
    """Utility class for handling audio preprocessing tasks such as format validation,
    conversion to WAV, and silence detection.

    Attributes:
        __out_dir (Path): The directory to output processed audio files.
        __supported_formats (Set[str]): Set of audio formats supported by FFmpeg for decoding.
    """

    def __init__(self):
        """Initializes the AudioHelper with the output directory and FFmpeg-supported decoding formats."""
        self.__out_dir = Path(config.AUDIO_OUT_DIR).resolve()
        self.__supported_formats = self.__get_ffmpeg_decoding_formats()

    def is_file_empty(self, infile: str) -> bool:
        """Checks whether a given audio file is considered empty.

        A file is considered empty if it is smaller than ~12KB or contains only silence.

        Args:
            infile (str): Path to the input file.

        Returns:
            bool: True if the file is empty or silent, False otherwise.
        """
        size_kb = os.path.getsize(infile) / 1024
        if size_kb <= 12:
            return True
        return self.__is_audio_empty(infile)

    def convert_audio_to_wav(self, infile: str) -> str:
        """Converts an input audio file to WAV format.

        Validates file format support based on FFmpeg's decoding capabilities,
        then uses `pydub` to perform the conversion.

        Args:
            infile (str): Path to the input audio file.

        Returns:
            str: Path to the generated WAV file.

        Raises:
            TranscriptionError: If the file format is unsupported or conversion fails.
        """
        if not self.__is_file_format_supported(infile):
            log.exception(f"File format of {infile} is not supported.")
            raise TranscriptionError(f"File format of {infile} is not supported.")

        outfile = (
            f"{self.__out_dir}{os.sep}{infile.split(os.sep)[-1].split('.')[0]}.wav"
        )

        log.info(f"Converting {infile} to WAV format as {outfile}")

        try:
            sound = pydub.AudioSegment.from_file(infile)
            sound.export(outfile, format="wav")
            return outfile
        except Exception as e:
            log.exception(f"Error during conversion of {infile} to WAV format: {e}")
            raise TranscriptionError(
                f"Error during conversion of {infile} to WAV format"
            )

    def __is_file_format_supported(self, filepath: str) -> bool:
        """Checks whether the given audio file's format is supported for decoding by FFmpeg.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            bool: True if the format is supported, False otherwise.
        """
        log.info(f"Checking if {filepath} is supported by ffmpeg")
        _, ext = os.path.splitext(filepath)
        ext = ext.lower().lstrip(".")

        return ext in self.__supported_formats

    @staticmethod
    def __is_audio_empty(
        infile: str, min_silence_len: int = 1000, silence_thresh: int = -50
    ) -> bool:
        """Check if an audio file is empty or contains only silence.

        Args:
            infile (str): Path to the input audio file.
            min_silence_len (int, optional): Minimum length of silence in milliseconds to consider. Defaults to 100.
            silence_thresh (int, optional): Silence threshold in dBFS. Default to -50.

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

    @staticmethod
    def __get_ffmpeg_decoding_formats() -> Set[str]:
        """Parses FFmpeg output to get a set of supported decoding formats.

        Returns:
            Set[str]: A set of lowercase format strings supported by FFmpeg for decoding.
        """
        log.info("Getting supported ffmpeg decoding formats")
        # run ffmpeg -formats to get supported formats
        result = subprocess.run(
            ["ffmpeg", "-formats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines = result.stdout.splitlines()

        # Parse stdout, the output is structured like this
        #
        # ffmpeg version, lib infos...
        # lib versions
        # File formats:
        # D. = Demuxing supported
        # .E = Muxing supported
        #  --
        # DE mp3 MP (MPEG audio layer 3)
        # ...
        #
        # So we want to start parsing the lines after the --
        # Extract the flags and the format
        # We only care about Decoding and then turning into a .wav
        # so we only add formats which have the D flag
        decoding_formats = set()
        start_parsing = False
        for line in lines:
            if line.strip().startswith("--"):
                start_parsing = True
                continue
            if start_parsing:
                flags = line[:3]
                parts = line[3:].strip().split()
                if not parts:
                    continue
                # for some reason this line exists
                # D  mov,mp4,m4a,3gp,3g2,mj2 QuickTime / MOV
                # so we also need to split the fmt by ,
                fmts = parts[0].split(",")
                for fmt in fmts:
                    if "D" in flags:
                        decoding_formats.add(fmt.lower())

        log.info("Supported formats will now be cached")
        return decoding_formats
