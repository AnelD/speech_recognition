import asyncio
import os
from pathlib import Path

import config


async def run_command_in_subprocess(command: str, cwd: str):
    """
    Run a command in a subprocess shell, it awaits the command execution.

    :param command: command to run
    :param cwd: directory to run the command in
    :return: None, prints stdout and stderr to console
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

    # Decode and print output.
    if stdout:
        print(f"[stdout]\n{stdout.decode()}")
    if stderr:
        print(f"[stderr]\n{stderr.decode()}")


async def get_audio(queue):
    """
    Turn a text message into an audio file.

    :param queue: asyncio.Queue from which to get the text
    :return: None
    """
    piper_dir = str(Path(config.PIPER_DIR.encode("unicode_escape").decode()))
    piper_command = rf"| .{os.sep}piper "
    voice = "-m " + config.VOICE_NAME
    output_path = "-d " + str(
        Path(config.GENERATE_AUDIO_DIR.encode("unicode_escape").decode())
    )
    loop = asyncio.get_event_loop()

    while True:
        text = await queue.get()
        input_text = f'echo "{text}" '
        print(input_text)
        command = input_text + piper_command + voice + output_path
        print(command)

        # Run the generation in separate thread to not block the event loop
        await loop.run_in_executor(
            None, run_command_in_subprocess, *(command, piper_dir)
        )
