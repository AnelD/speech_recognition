import asyncio
import glob
import json
import os
import shutil
from pathlib import Path

import pytest

import speech_recognition
from speech_recognition import LoggerHelper, main
from tests.mock_server import start_server_thread

log = LoggerHelper(__name__).get_logger()


@pytest.fixture(autouse=True, scope="session")
def change_test_dir():
    os.chdir(Path(__file__).parent)


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
    cwd = Path(os.getcwd())
    monkeypatch.setattr(
        speech_recognition.config, "GENERATE_AUDIO_DIR", str(cwd / "data/generated")
    )
    monkeypatch.setattr(speech_recognition.config, "AUDIO_IN_DIR", str(cwd / "data/in"))
    monkeypatch.setattr(speech_recognition.config, "AUDIO_OUT_DIR", str(cwd / "data/out"))

    # Start the mock server
    shutdown_event = asyncio.Event()
    received_queue = asyncio.Queue()
    server_thread = start_server_thread(
        out_queue=received_queue, shutdown_event=shutdown_event
    )

    # Start the application
    app_task = asyncio.create_task(main.main())

    # Wait for application to fully start
    register_message = await received_queue.get()
    assert register_message == "sp"

    try:
        await _send_and_assert_person_data(cwd, received_queue)

        await _send_and_assert_command_yes(cwd, received_queue)

    except Exception as e:
        log.error(e)
        pytest.fail(e)

    finally:
        # Clean up files after tests
        # Find all .wav files created during the test
        flac_files = glob.glob(str(Path("data/in").resolve() / "*.flac"))
        wav_files = glob.glob(str(Path("data/out").resolve() / "*.wav"))

        for file in flac_files:
            os.remove(file)
        for file in wav_files:
            os.remove(file)

        # Shutdown the app and server
        app_task.cancel()
        await app_task

        # Set shutdown event for server
        shutdown_event.set()
        server_thread.join(timeout=5)


async def _send_and_assert_person_data(cwd, received_queue):
    # Move a file to the watched directory
    src = Path(f"{cwd}/data/test_audios/person-test.flac").resolve()
    dest = Path(f"{cwd}/data/in/person-test.flac").resolve()
    # Delete the previous one if it exists so new event is sent out
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
                "phone_number": "0123 4567890",
                "sex": "M",
            }
            break
        else:
            pytest.fail("Received unknown message type")


async def _send_and_assert_command_yes(cwd, received_queue):
    expected = ["yes", "no"]
    for expect in expected:
        # Move a file to the watched directory
        src = Path(f"{cwd}/data/test_audios/command-test-{expect}.flac").resolve()
        dest = Path(f"{cwd}/data/in/command-test-{expect}.flac").resolve()
        # Delete the previous one if it exists so new event is sent out
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
                    "result": f"{expect.upper()}",
                }
                break
            else:
                pytest.fail("Received unknown message type")
