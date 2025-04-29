import asyncio
import time

import pytest

import speech_recognition
from speech_recognition import main
from tests.mock_server import start_mock_server


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
    monkeypatch.setattr(
        speech_recognition.config, "GENERATE_AUDIO_DIR", "../tests/data/generated"
    )

    # Start the mock server
    send_queue = asyncio.Queue()
    await asyncio.to_thread(start_mock_server, ("localhost", 8080, send_queue))

    # Start the application
    await asyncio.to_thread(main.main)

    time.sleep(5)
    await send_queue.put("Generate audio for this text")
