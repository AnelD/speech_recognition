import asyncio
import os
from pathlib import Path

from speech_recognition import config
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class TTSService:

    def __init__(self, queue: asyncio.Queue) -> None:
        self.__queue = queue

        # get piper directory
        self.__piper_dir = str(
            Path(config.PIPER_DIR.encode("unicode_escape").decode()).resolve()
        )

        # configure the command with values from a config file
        piper_command = rf"| .{os.sep}piper "
        voice = "-m " + config.VOICE_NAME
        output_path = " -d " + str(
            Path(config.GENERATE_AUDIO_DIR.encode("unicode_escape").decode()).resolve()
        )

        self.__prepared_command = piper_command + voice + output_path

    async def text_to_speech(self) -> None:
        """
        Turn a text message into an audio file.

        Args:
            queue (asyncio.Queue): asyncio.Queue from which to get the text
        Returns:
             None
        """

        # run forever waiting for inputs
        while True:
            text = await self.__queue.get()
            log.info(f"TTS starting with input: {text}")
            input_text = f'echo "{text}" '
            command = input_text + self.__prepared_command
            log.debug(f"Executing command: {command}")

            asyncio.create_task(self.__run_command_in_subprocess(command))

    async def __run_command_in_subprocess(self, command: str) -> None:
        """
        Run a command in a subprocess shell, it awaits the command execution.

        Args:
            command (str): command to run
            cwd (str): directory to run the command in
        Returns:
            None, logs stdout and stderr
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
        if stdout:
            log.info(f"[stdout]\n{stdout.decode()}")
        if stderr:
            log.info(f"[stderr]\n{stderr.decode()}")
