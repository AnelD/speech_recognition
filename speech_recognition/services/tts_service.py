import asyncio
import os
from pathlib import Path
from typing import Optional

from speech_recognition import config
from speech_recognition.exceptions.audio_generation_error import AudioGenerationError
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class TTSService:
    """Text-to-Speech (TTS) service for generating audio files from text input.

    This service uses a subprocess call to an external Piper TTS engine.
    It constructs the appropriate command using configuration values and executes it
    asynchronously to generate `.wav` audio files from the input text.

    Attributes:
        __piper_dir (str): Absolute path to the Piper directory.
        __prepared_command (str): Pre-built shell command for Piper with configured voice and output path.
    """

    def __init__(self) -> None:
        """Initializes the TTSService with the given Piper configuration."""

        self.__piper_dir = str(Path(config.PIPER_DIR).resolve())

        # configure the command with values from a config file
        piper_command = rf"| .{os.sep}piper "
        voice = "-m " + config.VOICE_NAME
        output_path = " -f " + str(Path(config.GENERATE_AUDIO_DIR).resolve())

        self.__prepared_command = piper_command + voice + output_path

    async def generate_audio(self, text) -> str:
        """Converts the given text into an audio file using the Piper TTS engine.

        Args:
            text (str): The input text to be converted to speech.

        Returns:
            str: Path to the generated audio file.

        Raises:
            AudioGenerationError: If the audio file could not be generated.
        """

        log.info(f"TTS starting with input: {text}")
        input_text = f'echo "{text}" '
        filename = "audio.wav"
        command = f"{input_text} {self.__prepared_command}{os.sep}{filename}"
        log.debug(f"Executing command: {command}")

        res = await self.__run_command_in_subprocess(command)
        if filename in res:
            return res
        else:
            log.error(f"Audio generation failed, received unexpected response: {res}")
            raise AudioGenerationError("Could not generate audio from text.")

    async def __run_command_in_subprocess(self, command: str) -> Optional[str]:
        """Executes a shell command asynchronously in the Piper directory.

        Args:
            command (str): The shell command to be executed.

        Returns:
            Optional[str]: The stdout output containing the generated filename, or None if generation failed.
        """
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            shell=True,
            cwd=self.__piper_dir,
        )

        # Wait for the process to complete.
        stdout, stderr = await proc.communicate()

        # Log the output
        if stderr:
            log.debug(f"[stderr]\n{stderr.decode()}")
        if stdout:
            log.debug(f"[stdout]\n{stdout.decode()}")
            return stdout.decode()

        return None
