import logging

import pytest

from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.services.asr_service import (
    ASRService,
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
    # Disables logging during tests
    logging.disable(logging.CRITICAL)


def test_asrservice_load_model(mocker):
    # Mock the pipeline so we don't have to load a real model
    mock_pipeline = mocker.patch("speech_recognition.services.asr_service.pipeline")
    service = ASRService()

    mock_pipeline.assert_called_once()
    assert service is not None


def test_asrservice_transcribe_success(mocker, dummy_audio_path, dummy_wav_path):
    # Mock the things as we don't want to run a real model
    mock_model = mocker.Mock(return_value={"text": "Hello world"})
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
    # Mock it with a return value that results in failure
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
    # Mock the model throwing an exception
    mock_model = mocker.Mock(side_effect=Exception("Inference crashed"))
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
