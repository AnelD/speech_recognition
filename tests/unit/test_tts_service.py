import asyncio
import logging

import pytest

import speech_recognition
import speech_recognition.services.tts_service


@pytest.fixture(autouse=True)
def disable_logging():
    # Disables logging during tests
    logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
async def test_run_command_in_subprocess_is_called(mocker, monkeypatch):
    # Mock the piper directory
    monkeypatch.setattr(speech_recognition.config, "PIPER_DIR", "/mock/piper")
    # Mock the creation of a subprocess shell
    mock_proc = mocker.AsyncMock()
    mock_proc.communicate.return_value = (b"Hello stdout", b"Hello stderr")
    mock_sub_shell = mocker.patch(
        "speech_recognition.services.tts_service.asyncio.create_subprocess_shell",
        return_value=mock_proc,
    )

    # Create an instance with a dummy queue
    service = speech_recognition.services.tts_service.TTSService(
        asyncio.Queue(), mocker.AsyncMock()
    )

    # Run it
    await service._TTSService__run_command_in_subprocess("echo hi")

    # Assert it was called correctly
    mock_proc.communicate.assert_called_once()
    mock_sub_shell.assert_called_once_with(
        "echo hi",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
        cwd="/mock/piper",
    )


@pytest.mark.asyncio
async def test_text_to_speech_creates_correct_command(mocker, monkeypatch):
    # Mock config values
    monkeypatch.setattr(speech_recognition.config, "PIPER_DIR", "/mock/piper")
    monkeypatch.setattr(speech_recognition.config, "VOICE_NAME", "mock_voice")
    monkeypatch.setattr(speech_recognition.config, "GENERATE_AUDIO_DIR", "/mock/output")

    # Create queue and put a message
    queue = asyncio.Queue()
    await queue.put("hello world")

    # Create the service instance
    service = speech_recognition.services.tts_service.TTSService(
        queue, mocker.AsyncMock()
    )

    # Replace real _run_command_in_subprocess so we can look at the args it was called with
    mock_run = mocker.AsyncMock()
    mocker.patch.object(service, "_TTSService__run_command_in_subprocess", mock_run)

    # Await for the task to start
    task = asyncio.create_task(service.text_to_speech())
    await asyncio.sleep(0.1)

    # We don't need it to actually do anything
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Assert the things
    assert mock_run.call_count == 1
    called_command = mock_run.call_args.args[0]
    assert "hello world" in called_command
    assert ".\\piper" in called_command or "./piper" in called_command
    assert "-m mock_voice" in called_command
    assert "-f /mock/output/" in called_command
