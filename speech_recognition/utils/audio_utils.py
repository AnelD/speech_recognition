import os

import pydub
from pydub.silence import detect_nonsilent

from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.utils.logger_helper import LoggerHelper

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
    if size_kb <= 12:
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
