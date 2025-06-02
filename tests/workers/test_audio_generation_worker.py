import asyncio
import os

import pytest

from speech_recognition.exceptions.audio_generation_error import AudioGenerationError
from speech_recognition.workers.audio_generation_worker import AudioGenerationWorker


@pytest.fixture
def text_queue():
    return asyncio.Queue()


@pytest.fixture
def client(mocker):
    """
    Create a simple mock client that records messages sent.
    """
    client_mock = mocker.AsyncMock()
    client_mock.messages = list()

    async def send_message(message):
        client_mock.messages.append(message)

    client_mock.send_message = send_message
    return client_mock


@pytest.fixture
def tts_service(mocker):
    """
    Create a mock TTS service.
    """
    return mocker.Mock()


@pytest.fixture
def worker(text_queue, tts_service, client):
    """
    Create an AudioGenerationWorker instance with mocked dependencies.
    """
    return AudioGenerationWorker(text_queue, tts_service, client)


@pytest.mark.asyncio
async def test_successful_audio_generation(
    mocker, worker, text_queue, client, tts_service
):
    """
    Test that a valid request to generate audio leads to a success message.
    """
    # Prepare a fake request payload (could be text or any dict as expected by tts_service.generate_audio)
    request = "Some text to generate audio"
    await text_queue.put(request)

    # Configure the tts_service.generate_audio to return a fake file path.
    fake_file_path = os.path.join("fake", "path", "audio.wav")
    tts_service.generate_audio.return_value = asyncio.Future()
    tts_service.generate_audio.return_value.set_result(fake_file_path)

    # Run the worker's do_work loop for a short time.
    task = asyncio.create_task(worker.do_work())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify that the client received a success message.
    assert len(client.messages) == 1
    msg = client.messages[0]
    assert msg["type"] == "GENERATE_AUDIO_SUCCESS"
    # The message should include only the file name portion.
    expected_filename = fake_file_path.split(os.sep)[-1]
    assert expected_filename in msg["message"]["text"]


@pytest.mark.asyncio
async def test_audio_generation_error(mocker, worker, text_queue, client, tts_service):
    """
    Test that when tts_service.generate_audio raises an AudioGenerationError,
    the worker sends an error message.
    """
    # Prepare a fake request.
    request = "Some text to generate audio"
    await text_queue.put(request)

    # Configure generate_audio to raise an AudioGenerationError.
    error_message = "Failed to generate audio"
    tts_service.generate_audio.side_effect = AudioGenerationError(error_message)

    # Run the worker's do_work loop for a short time.
    task = asyncio.create_task(worker.do_work())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Verify that an error message has been sent.
    assert len(client.messages) == 1
    msg = client.messages[0]
    assert msg["type"] == "GENERATE_AUDIO_ERROR"
    assert error_message in msg["message"]["text"]
