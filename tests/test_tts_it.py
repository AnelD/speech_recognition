import asyncio
import glob
import os
from pathlib import Path

import pytest

import speech_recognition
from speech_recognition import main
from tests.mock_server import start_server_thread


@pytest.fixture(autouse=True, scope="session")
def change_test_dir():
    os.chdir(Path(__file__).parent)


@pytest.mark.asyncio
async def test_tts_it(monkeypatch):
    """Integration test for the full text to speech workflow"""

    # Set model config to the smallest ones for faster startup
    monkeypatch.setattr(
        speech_recognition.config, "ASR_MODEL_NAME", "openai/whisper-tiny"
    )
    monkeypatch.setattr(
        speech_recognition.config, "LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"
    )
    # Set websocket config to point to mockserver
    monkeypatch.setattr(
        speech_recognition.config, "WEBSOCKET_URI", "ws://localhost:8080"
    )
    # generate audio in test folder
    cwd = os.getcwd()
    monkeypatch.setattr(
        speech_recognition.config, "GENERATE_AUDIO_DIR", cwd + "/data/generated"
    )

    # Start the mock server
    shutdown_event = asyncio.Event()
    send_queue = asyncio.Queue()
    server_thread = start_server_thread(
        in_queue=send_queue, shutdown_event=shutdown_event
    )

    await asyncio.sleep(1)

    # Start the application
    asyncio.create_task(main.main())

    # Send a message from the mockserver
    await asyncio.sleep(1)
    await send_queue.put("Generate some audio for this message")
    await asyncio.sleep(1)
    await send_queue.put(None)
    await asyncio.sleep(5)

    # Find all .wav files created during the test
    output_dir = "../tests/data/generated"
    wav_files = glob.glob(os.path.join(output_dir, "*.wav"))

    # Assert at least one .wav file was created
    assert len(wav_files) > 0, "No .wav files were generated"

    # Clean up generated files
    for file in wav_files:
        os.remove(file)

    # Shutdown the app and server
    # Cancel all tasks before event loop closes
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    # Set shutdown event for server
    shutdown_event.set()
    server_thread.join(timeout=5)
