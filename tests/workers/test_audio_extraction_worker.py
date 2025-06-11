import asyncio
import os

import pytest

from speech_recognition.exceptions.llm_processing_error import LLMProcessingError
from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.workers.audio_extraction_worker import AudioExtractionWorker


@pytest.fixture
def speech_queue():
    return asyncio.Queue()


@pytest.fixture
def client(mocker):
    """
    A mock client that records messages sent via send_message.
    """
    client_mock = mocker.AsyncMock()
    client_mock.messages = list()

    # We'll define a simple async function on the mock:
    async def send_message(message):
        client_mock.messages.append(message)

    client_mock.send_message = send_message
    return client_mock


@pytest.fixture
def asr_service(mocker):
    """
    Return a mock for the asr_service.
    """
    asr = mocker.Mock()
    # By default, transcribe returns a simple string based on the file name
    asr.transcribe.return_value = None  # This will be overridden in tests accordingly.
    return asr


@pytest.fixture
def llm_service(mocker):
    """
    Return a mock for the llm_service.
    """
    llm = mocker.Mock()
    # By default, generate_json_response returns a dummy json
    llm.generate_json_response.return_value = None  # This will be set in tests.
    return llm


@pytest.fixture
def worker(speech_queue, asr_service, llm_service, client):
    """
    Create an AudioExtractionWorker instance with mocked services.
    """
    return AudioExtractionWorker(speech_queue, asr_service, llm_service, client)


def run_one_iteration(worker, loop):
    """
    Helper that runs one iteration of the worker.do_work loop.
    Since do_work is an infinite loop waiting for queue items,
    we schedule it as a task, wait a short while, and then cancel.
    """

    async def inner():
        task = asyncio.create_task(worker.do_work())
        # Allow the worker to process the request(s)
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    loop.run_until_complete(inner())


# ---------------------------
# Tests
# ---------------------------


@pytest.mark.asyncio
async def test_successful_audio_extraction(
    mocker, worker, speech_queue, client, asr_service, llm_service
):
    """
    Test the successful processing of an audio extraction request.
    """
    file_path = os.path.join("path", "to", "audio.wav")
    request = {"file": file_path, "req_type": "VALID_REQUEST"}
    await speech_queue.put(request)

    # Patch AudioHelper that might be used in asr_service.
    # Adjust the patch target according to your project structure.
    mock_audio_helper = mocker.patch(
        "speech_recognition.services.asr_service.AudioHelper"
    ).return_value
    mock_audio_helper.convert_audio_to_wav.return_value = None
    mock_audio_helper.is_file_empty.return_value = False

    # Configure the mocked asr_service.transcribe method.
    asr_service.transcribe.side_effect = lambda file: f"mocked transcription of {file}"

    # Configure the mocked llm_service.generate_json_response method.
    llm_service.generate_json_response.side_effect = lambda text, req_type: {
        "transcribed": text,
        "req_type": req_type,
    }

    # Run one iteration of the worker loop.
    task = asyncio.create_task(worker.do_work())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # The worker should send two messages: one starting and one success.
    assert len(client.messages) == 2

    starting_msg = client.messages[0]
    success_msg = client.messages[1]

    file_name = file_path.split(os.sep)[-1]
    assert starting_msg["type"] == "EXTRACT_DATA_FROM_AUDIO_STARTING"
    assert file_name in starting_msg["message"]["text"]

    # Verify success message content
    assert success_msg["type"] == "EXTRACT_DATA_FROM_AUDIO_SUCCESS"
    result = success_msg["message"]["text"]
    expected_transcription = f"mocked transcription of {file_path}"
    assert result["transcribed"] == expected_transcription
    assert result["req_type"] == "VALID_REQUEST"


@pytest.mark.asyncio
async def test_bad_request(worker, speech_queue, client):
    """
    Test that a request with req_type "BAD_REQUEST" sends an error message.
    """
    file_path = os.path.join("path", "to", "audio.wav")
    request = {"file": file_path, "req_type": "BAD_REQUEST"}
    await speech_queue.put(request)

    task = asyncio.create_task(worker.do_work())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Expect only an error message to be sent.
    assert len(client.messages) == 1
    error_msg = client.messages[0]
    assert error_msg["type"] == "EXTRACT_DATA_FROM_AUDIO_ERROR"
    # Check that the error message includes the file info.
    assert file_path in error_msg["message"]["text"]


@pytest.mark.asyncio
async def test_transcription_error(mocker, worker, speech_queue, client, asr_service):
    """
    Test that when asr_service.transcribe raises a TranscriptionError,
    an error message is sent.
    """
    file_path = os.path.join("path", "to", "audio.wav")
    request = {"file": file_path, "req_type": "VALID_REQUEST"}
    await speech_queue.put(request)

    # Configure transcribe to raise TranscriptionError.
    asr_service.transcribe.side_effect = TranscriptionError("Transcription failed")

    task = asyncio.create_task(worker.do_work())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Check that an error message is sent.
    error_msgs = [
        msg for msg in client.messages if msg["type"] == "EXTRACT_DATA_FROM_AUDIO_ERROR"
    ]
    assert error_msgs, "Expected an error message upon transcription failure"
    assert "Transcription failed" in error_msgs[0]["message"]["text"]


@pytest.mark.asyncio
async def test_llm_processing_error(mocker, worker, speech_queue, client, llm_service):
    """
    Test that when llm_service.generate_json_response raises an LLMProcessingError,
    an error message is sent.
    """
    file_path = os.path.join("path", "to", "audio.wav")
    request = {"file": file_path, "req_type": "VALID_REQUEST"}
    await speech_queue.put(request)

    # Configure generate_json_response to raise LLMProcessingError.
    llm_service.generate_json_response.side_effect = LLMProcessingError(
        "LLM processing failed"
    )

    task = asyncio.create_task(worker.do_work())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Check that an error message is sent.
    error_msgs = [
        msg for msg in client.messages if msg["type"] == "EXTRACT_DATA_FROM_AUDIO_ERROR"
    ]
    assert error_msgs, "Expected an error message upon LLM processing failure"
    assert "LLM processing failed" in error_msgs[0]["message"]["text"]
