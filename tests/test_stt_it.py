import asyncio
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from websocket_server import WebsocketServer

import speech_recognition
from speech_recognition import main, LoggerHelper
from tests.mock_server import start_mock_server, serve_mock

log = LoggerHelper(__name__).get_logger()


@pytest.fixture(scope="module")
def mock_ws_server():
    received_messages = []
    message_event = asyncio.Event()

    async def message_received(client, server, message):
        print(f"Received message: {message}")
        received_messages.append(message)
        message_event.set()  # Notify a waiting thread

    # Create the server
    server = WebsocketServer(host="127.0.0.1", port=8080, loglevel=1)
    server.set_fn_message_received(message_received)

    # Running the server in a background thread
    def start_server():
        server.run_forever()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(0.5)  # Give it time to start

    class ServerHelper:
        def __init__(self):
            self.received_messages = received_messages
            self.message_event = message_event

        async def get_messages(self):
            return list(self.received_messages)

        async def wait_for_message(self, match=None, timeout=5):
            """Wait for a message matching 'match'."""
            deadline = time.time() + timeout
            while time.time() < deadline:
                for msg in self.received_messages:
                    if match is None or match in msg:
                        return msg
                await asyncio.sleep(0.1)  # Non-blocking sleep
            raise TimeoutError(f"Timeout waiting for message matching: {match}")

    yield ServerHelper()

    server.shutdown()


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
    # server_thread = start_server_in_thread(received_queue)
    # server_task = asyncio.create_task(start_mock_server(received_queue))
    # server_task2 = start_server2(received_queue)
    # server_task2.start()
    # server3 = start_mock_ws_server()
    log.error("Waiting before server start")
    await asyncio.sleep(5)
    log.error("Starting server")
    server3_thread = start_server3(received_queue)

    # await asyncio.sleep(1)
    # server3_thread.start()

    log.error("Waiting after server start")
    await asyncio.sleep(5)

    log.error("Starting app")
    # Start the application
    app_task = asyncio.create_task(main.main())
    log.error("app_task started")
    # Wait for application to fully start
    register_message = await received_queue.get()
    log.info(register_message)

    # Move a file to the watched directory
    src = Path("../tests/data/test_audios/person-Test.flac").resolve()
    dest = Path("../tests/data/in/person-Test.flac").resolve()
    await asyncio.sleep(5)
    if os.path.exists(dest):
        os.remove(dest)
    shutil.copy(src, dest)

    # msg = await mock_ws_server.wait_for_message("REGISTER_CLIENT", timeout=15)
    while True:
        resp = await received_queue.get()
        json_resp = json.loads(resp)
        log.debug(json_resp)
        if json_resp["type"] == "EXTRACT_DATA_FROM_AUDIO_STARTING":
            continue
        if json_resp["type"] == "EXTRACT_DATA_FROM_AUDIO_SUCCESS":
            break
        else:
            pytest.fail("Received unknown message type")

    # shut down
    # server_task.cancel()
    # with pytest.raises(asyncio.CancelledError):
    # await server_task

    app_task.cancel()


def start_server_in_thread(queue):
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run_server():
            server = await start_mock_server(out_queue=queue)
            while True:
                await asyncio.sleep(1)  # Keep the server alive

        loop.run_until_complete(run_server())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def start_server2(queue):
    def thread_target():
        async def run_server():
            server = await start_mock_server(out_queue=queue)
            # await server.wait_closed()  # Keeps server alive
            while True:
                await asyncio.sleep(1)  # Keeps server alive

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server())

    thread = threading.Thread(target=thread_target, daemon=True)
    return thread


def message_received(client, server, message):
    log.debug(f"Received: {message}")


def start_mock_ws_server():
    server = WebsocketServer(host="127.0.0.1", port=8080)
    server.set_fn_message_received(message_received)

    thread = threading.Thread(target=server.run_forever, daemon=True)
    thread.start()
    return server


def start_server3(queue):
    thread = threading.Thread(
        target=serve_mock,
        kwargs={"out_queue": queue, "in_queue": asyncio.Queue()},
        daemon=True,
    )
    thread.start()
    return thread
