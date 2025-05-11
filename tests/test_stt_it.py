import asyncio
import json
import os
import shutil
import threading
from pathlib import Path

import pytest

import speech_recognition
from speech_recognition import main, LoggerHelper
from tests.mock_server import serve_mock

log = LoggerHelper(__name__).get_logger()


def start_server(queue):
    thread = threading.Thread(
        target=serve_mock,
        kwargs={"out_queue": queue, "in_queue": asyncio.Queue()},
        daemon=True,
    )
    thread.start()
    return thread


@pytest.mark.asyncio
async def test_stt_it(monkeypatch):
    """Integration test for the full text to speech workflow"""

    # Set model config
    monkeypatch.setattr(
        speech_recognition.config, "ASR_MODEL_NAME", "openai/whisper-large-v3-turbo"
    )
    monkeypatch.setattr(
        speech_recognition.config, "LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"
    )
    # Set websocket config to point to mockserver
    monkeypatch.setattr(
        speech_recognition.config, "WEBSOCKET_URI", "ws://localhost:8080"
    )
    # Set it to watch the test folders
    monkeypatch.setattr(speech_recognition.config, "AUDIO_IN_DIR", "../tests/data/in")
    monkeypatch.setattr(speech_recognition.config, "AUDIO_OUT_DIR", "../tests/data/out")

    # Start the mock server
    received_queue = asyncio.Queue()
    server_thread = start_server(received_queue)

    # Start the application
    app_task = asyncio.create_task(main.main())

    # Wait for application to fully start
    register_message = await received_queue.get()
    assert register_message == "sp"

    # Move a file to the watched directory
    src = Path("../tests/data/test_audios/person-Test.flac").resolve()
    dest = Path("../tests/data/in/person-Test.flac").resolve()
    await asyncio.sleep(5)
    if os.path.exists(dest):
        os.remove(dest)
    shutil.copy(src, dest)

    while True:
        resp = await received_queue.get()
        json_resp = json.loads(resp)
        log.debug(json_resp)
        if json_resp["type"] == "EXTRACT_DATA_FROM_AUDIO_STARTING":
            continue
        if json_resp["type"] == "EXTRACT_DATA_FROM_AUDIO_SUCCESS":
            assert json_resp["message"]["text"] == {
                "date_of_birth": "1995-05-15",
                "email_address": "maxilianemustermann.gmail.com",
                "firstname": "Maximiliane",
                "lastname": "Mustermann",
                "phone_number": "0123 45 67 890",
                "sex": "M",
            }
            break
        else:
            pytest.fail("Received unknown message type")

    # Cancel all tasks before event loop closes
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Shutdown the app and server
    app_task.cancel()
    try:
        await app_task
    except asyncio.CancelledError:
        pass
    server_thread.join(timeout=5)
