import asyncio

import pytest

import speech_recognition
import speech_recognition.services.tts_service


@pytest.mark.asyncio
async def test_run_command_in_subprocess_logs_stdout_and_stderr(mocker):
    # Mock the creation of a subprocess shell
    mock_proc = mocker.AsyncMock()
    mock_proc.communicate.return_value = (b"Hello stdout", b"Hello stderr")
    mock_sub_shell = mocker.patch(
        "speech_recognition.services.tts_service.asyncio.create_subprocess_shell",
        return_value=mock_proc,
    )

    # Run it
    await speech_recognition.services.tts_service.run_command_in_subprocess(
        "echo hi", cwd="/tmp"
    )

    # Assert it was called correctly
    mock_proc.communicate.assert_called_once()
    mock_sub_shell.assert_called_once_with(
        "echo hi",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
        cwd="/tmp",
    )


@pytest.mark.asyncio
async def test_text_to_speech_creates_correct_command(mocker, monkeypatch):
    # Mock config values
    monkeypatch.setattr(speech_recognition.config, "PIPER_DIR", "/mock/piper")
    monkeypatch.setattr(speech_recognition.config, "VOICE_NAME", "mock_voice")
    monkeypatch.setattr(speech_recognition.config, "GENERATE_AUDIO_DIR", "/mock/output")

    # Replace real run_command... so we can look at the args it was called with
    mock_run = mocker.AsyncMock()
    monkeypatch.setattr(
        speech_recognition.services.tts_service, "run_command_in_subprocess", mock_run
    )

    # Create queue and await for the task to start
    queue = asyncio.Queue()
    await queue.put("hello world")
    task = asyncio.create_task(
        speech_recognition.services.tts_service.text_to_speech(queue)
    )
    await asyncio.sleep(0.1)

    # We don't need it to actually do anything
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Assert the things
    assert mock_run.call_count == 1
    called_command = mock_run.call_args[0][0]
    called_cwd = mock_run.call_args[1]["cwd"]
    assert "hello world" in called_command
    assert ".\\piper" in called_command or "./piper" in called_command
    assert "-m mock_voice" in called_command
    assert "-d" in called_command
    assert "/mock/piper" in called_cwd
