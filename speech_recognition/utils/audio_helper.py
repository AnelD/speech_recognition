import os
import subprocess
from typing import Set

import pydub
from pydub.silence import detect_nonsilent

from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class AudioHelper:
    def __init__(self):
        self._supported_formats = self._get_ffmpeg_decoding_formats()

    def is_file_empty(self, infile: str) -> bool:
        """Check if a file is empty based on file size or audio content.

        Args:
            infile (str): Path to the input file.

        Returns:
            bool: True if the file is empty or contains only silence, False otherwise.
        """
        size_kb = os.path.getsize(infile) / 1024
        if size_kb <= 12:
            return True
        return self._is_audio_empty(infile)

    def convert_audio_to_wav(self, infile: str, outfile: str) -> None:
        """Convert an input audio file to WAV format.

        Args:
            infile (str): Path to the input audio file.
            outfile (str): Path to save the output WAV file.

        Returns:
            None
        """
        if not self._is_file_format_supported(infile):
            log.exception(f"File format of {infile} is not supported.")
            raise TranscriptionError(f"File format of {infile} is not supported.")

        log.info(f"Converting {infile} to WAV format as {outfile}")

        try:
            sound = pydub.AudioSegment.from_file(infile)
            sound.export(outfile, format="wav")
        except Exception as e:
            log.exception(f"Error during conversion of {infile} to WAV format: {e}")
            raise TranscriptionError(
                f"Error during conversion of {infile} to WAV format"
            )

    def _is_file_format_supported(self, filepath: str) -> bool:
        """
        Checks if the file extension of `filepath` is in the list of formats
        supported for decoding by ffmpeg.
        """
        log.info(f"Checking if {filepath} is supported by ffmpeg")
        _, ext = os.path.splitext(filepath)
        ext = ext.lower().lstrip(".")

        return ext in self._supported_formats

    @staticmethod
    def _is_audio_empty(
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

    @staticmethod
    def _get_ffmpeg_decoding_formats() -> Set[str]:
        """
        Cached function to get supported ffmpeg decoding formats.
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


if __name__ == "__main__":
    audio_helper = AudioHelper()
    print(audio_helper._supported_formats)
    print(audio_helper._get_ffmpeg_decoding_formats())
    print(audio_helper._get_ffmpeg_decoding_formats())
    print(audio_helper._is_file_format_supported("test.mp3"))
    print(audio_helper._is_file_format_supported("test.m4a"))
    print(audio_helper._is_file_format_supported("test.m4b"))
    audio_helper.convert_audio_to_wav("test.m4b", "test.wav")
