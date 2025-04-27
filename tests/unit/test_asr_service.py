import logging
from unittest.mock import MagicMock

import pytest

from speech_recognition.services.asr_service import (
    is_audio_empty,
    is_file_empty,
    convert_audio_to_wav,
    ASRService,
    TranscriptionError,
)


# Fixtures
@pytest.fixture
def dummy_audio_path(tmp_path):
    return tmp_path / "dummy_audio.mp3"


@pytest.fixture
def dummy_wav_path(tmp_path):
    return tmp_path / "dummy_audio.wav"


@pytest.fixture
def mock_from_file(mocker):
    mocker.patch("speech_recognition.services.asr_service.pydub.AudioSegment.from_file")


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.CRITICAL)  # Disables logging during tests


# --- is_audio_empty tests ---
@pytest.mark.parametrize("test_input,expected", [([], True), ([1, 2, 3], False)])
def test_is_audio_empty_returns_true(
    mocker, mock_from_file, dummy_audio_path, test_input, expected
):
    mocker.patch(
        "speech_recognition.services.asr_service.detect_nonsilent",
        return_value=test_input,
    )
    result = is_audio_empty(str(dummy_audio_path))
    assert result is expected


# --- is_file_empty tests ---
def test_is_file_empty_small_file(mocker, dummy_audio_path):
    mocker.patch(
        "speech_recognition.services.asr_service.os.path.getsize", return_value=1024 * 3
    )
    mock_is_audio_empty = mocker.patch(
        "speech_recognition.services.asr_service.is_audio_empty"
    )
    result = is_file_empty(str(dummy_audio_path))
    assert result is True
    mock_is_audio_empty.assert_not_called()


def test_is_file_empty_large_file_silent(mocker, dummy_audio_path):
    mocker.patch(
        "speech_recognition.services.asr_service.os.path.getsize",
        return_value=1024 * 10,
    )
    mock_is_audio_empty = mocker.patch(
        "speech_recognition.services.asr_service.is_audio_empty", return_value=True
    )
    result = is_file_empty(str(dummy_audio_path))
    assert result is True
    mock_is_audio_empty.assert_called_once()


def test_is_file_empty_large_file_non_silent(mocker, dummy_audio_path):
    mocker.patch(
        "speech_recognition.services.asr_service.os.path.getsize",
        return_value=1024 * 10,
    )
    mock_is_audio_empty = mocker.patch(
        "speech_recognition.services.asr_service.is_audio_empty", return_value=False
    )
    result = is_file_empty(str(dummy_audio_path))
    assert result is False
    mock_is_audio_empty.assert_called_once()


# --- convert_audio_to_wav tests ---
def test_convert_audio_to_wav_success(mocker, dummy_audio_path, dummy_wav_path):
    mock_audio = mocker.MagicMock()
    mocker.patch(
        "speech_recognition.services.asr_service.pydub.AudioSegment.from_file",
        return_value=mock_audio,
    )

    convert_audio_to_wav(str(dummy_audio_path), str(dummy_wav_path))
    mock_audio.export.assert_called_once_with(str(dummy_wav_path), format="wav")


def test_convert_audio_to_wav_failure(mocker, dummy_audio_path, dummy_wav_path):
    mocker.patch(
        "speech_recognition.services.asr_service.pydub.AudioSegment.from_file",
        side_effect=Exception("Something went wrong"),
    )

    with pytest.raises(TranscriptionError):
        convert_audio_to_wav(str(dummy_audio_path), str(dummy_wav_path))


# --- ASRService tests ---
def test_asrservice_load_model(mocker):
    mock_pipeline = mocker.patch("speech_recognition.services.asr_service.pipeline")
    service = ASRService()
    mock_pipeline.assert_called_once()
    assert service is not None


def test_asrservice_transcribe_success(mocker, dummy_audio_path, dummy_wav_path):
    mock_model = MagicMock(return_value={"text": "Hello world"})
    mock_convert = mocker.patch(
        "speech_recognition.services.asr_service.convert_audio_to_wav"
    )
    mocker.patch(
        "speech_recognition.services.asr_service.is_file_empty", return_value=False
    )
    mocker.patch(
        "speech_recognition.services.asr_service.pipeline", return_value=mock_model
    )

    service = ASRService()
    text = service.transcribe(str(dummy_audio_path), str(dummy_wav_path))
    assert text == "Hello world"
    mock_convert.assert_called_once()


def test_asrservice_transcribe_empty_file_raises(
    mocker, dummy_audio_path, dummy_wav_path
):
    mocker.patch(
        "speech_recognition.services.asr_service.is_file_empty", return_value=True
    )
    mocker.patch(
        "speech_recognition.services.asr_service.pipeline",
    )
    service = ASRService()

    with pytest.raises(TranscriptionError, match="empty or contains only silence"):
        service.transcribe(str(dummy_audio_path), str(dummy_wav_path))


def test_asrservice_transcribe_exception_during_inference(
    mocker, dummy_audio_path, dummy_wav_path
):
    mock_model = MagicMock(side_effect=Exception("Inference crashed"))
    mocker.patch("speech_recognition.services.asr_service.convert_audio_to_wav")
    mocker.patch(
        "speech_recognition.services.asr_service.is_file_empty", return_value=False
    )
    mocker.patch(
        "speech_recognition.services.asr_service.pipeline", return_value=mock_model
    )

    service = ASRService()

    with pytest.raises(TranscriptionError, match="Error while transcribing"):
        service.transcribe(str(dummy_audio_path), str(dummy_wav_path))
