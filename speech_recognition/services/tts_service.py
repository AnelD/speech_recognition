import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from speech_recognition import config, WebSocketClient
from speech_recognition.exceptions.audio_generation_error import AudioGenerationError
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class TTSService:

    def __init__(self, queue: asyncio.Queue, client: WebSocketClient) -> None:
        self.__queue = queue
        self.__client = client

        # get piper directory
        self.__piper_dir = str(
            Path(config.PIPER_DIR.encode("unicode_escape").decode()).resolve()
        )

        # configure the command with values from a config file
        piper_command = rf"| .{os.sep}piper "
        voice = "-m " + config.VOICE_NAME
        output_path = " -f " + str(
            Path(config.GENERATE_AUDIO_DIR.encode("unicode_escape").decode()).resolve()
        )

        self.__prepared_command = piper_command + voice + output_path

    async def generate_audio(self, text) -> str:
        """
        Turn a text message into an audio file.

        Returns:
             str : Path to the generated audio file.
        """

        # run forever waiting for inputs
        log.info(f"TTS starting with input: {text}")
        input_text = f'echo "{text}" '
        filename = datetime.now().strftime("%Y_%m_%d_%H:%M:%S") + ".wav"
        command = f"{input_text} {self.__prepared_command}{os.sep}{filename}"
        log.debug(f"Executing command: {command}")

        res = await self.__run_command_in_subprocess(command)
        if filename in res:
            return res
        else:
            log.error(f"Audio generation failed, received unexpected response: {res}")
            raise AudioGenerationError("Could not generate audio from text.")

    async def __run_command_in_subprocess(self, command: str) -> Optional[str]:
        """
        Run a command in a subprocess shell, it awaits the command execution.

        Args:
            command (str): command to run
            cwd (str): directory to run the command in
        Returns:
            Optional[str]: filename of the created audio file if successful, None otherwise
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
