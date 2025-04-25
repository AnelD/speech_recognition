import asyncio
import logging
import os
from pathlib import Path

import config
from logger_helper import LoggerHelper

log = LoggerHelper(__name__, log_level=logging.DEBUG).get_logger()


async def run_command_in_subprocess(command: str, cwd: str):
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
        cwd=cwd,
    )

    # Wait for the process to complete.
    stdout, stderr = await proc.communicate()

    # Log the output
    if stdout:
        log.info(f"[stdout]\n{stdout.decode()}")
    if stderr:
        log.info(f"[stderr]\n{stderr.decode()}")


async def text_to_speech(queue: asyncio.Queue):
    """
    Turn a text message into an audio file.

    Args:
        queue (asyncio.Queue): asyncio.Queue from which to get the text
    Returns:
         None
    """

    # configure the command with values from a config file
    piper_dir = str(Path(config.PIPER_DIR.encode("unicode_escape").decode()))
    piper_command = rf"| .{os.sep}piper "
    voice = "-m " + config.VOICE_NAME
    output_path = " -d " + str(
        Path(config.GENERATE_AUDIO_DIR.encode("unicode_escape").decode())
    )
    command_without_input = piper_command + voice + output_path

    # run forever waiting for inputs
    while True:
        text = await queue.get()
        log.info(f"TTS starting with input: {text}")
        input_text = f'echo "{text}" '
        command = input_text + command_without_input
        log.debug(f"Executing command: {command}")

        asyncio.create_task(run_command_in_subprocess(command, cwd=piper_dir))
