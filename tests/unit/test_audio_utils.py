import logging

import pytest

from speech_recognition.exceptions.transcription_error import TranscriptionError
from speech_recognition.utils.audio_helper import AudioHelper


# Fixtures
@pytest.fixture
def dummy_audio_path(tmp_path):
    return tmp_path / "dummy_audio.mp3"


@pytest.fixture
def dummy_wav_path(tmp_path):
    return tmp_path / "dummy_audio.wav"


@pytest.fixture
def mock_from_file(mocker):
    mocker.patch("speech_recognition.utils.audio_helper.pydub.AudioSegment.from_file")


@pytest.fixture(autouse=True)
def disable_logging():
    # Disables logging during tests
    logging.disable(logging.CRITICAL)


# --- is_audio_empty tests ---
@pytest.mark.parametrize("test_input,expected", [([], True), ([1, 2, 3], False)])
def test_is_audio_empty(mocker, mock_from_file, dummy_audio_path, test_input, expected):
    # Mock with parametrized values
    mocker.patch(
        "speech_recognition.utils.audio_helper.detect_nonsilent",
        return_value=test_input,
    )
    helper = AudioHelper()
    result = helper._is_audio_empty(str(dummy_audio_path))
    assert result is expected


def test_is_file_empty_small_file(mocker, dummy_audio_path):
    # Mock a file being passed that is smaller than the threshold
    mocker.patch(
        "speech_recognition.utils.audio_helper.os.path.getsize",
        return_value=1024 * 3,
    )
    mock_is_audio_empty = mocker.patch(
        "speech_recognition.utils.audio_helper.AudioHelper._is_audio_empty"
    )
    helper = AudioHelper()
    result = helper.is_file_empty(str(dummy_audio_path))
    assert result is True
    # Assert that it actually returned because of the file size
    mock_is_audio_empty.assert_not_called()


def test_is_file_empty_large_file_silent(mocker, dummy_audio_path):
    # Mock a file being bigger than the threshold
    mocker.patch(
        "speech_recognition.utils.audio_helper.os.path.getsize",
        return_value=1024 * 15,
    )
    mock_is_audio_empty = mocker.patch(
        "speech_recognition.utils.audio_helper.AudioHelper._is_audio_empty",
        return_value=True,
    )
    helper = AudioHelper()
    result = helper.is_file_empty(str(dummy_audio_path))
    assert result is True
    mock_is_audio_empty.assert_called_once()


def test_is_file_empty_large_file_non_silent(mocker, dummy_audio_path):
    mocker.patch(
        "speech_recognition.utils.audio_helper.os.path.getsize",
        return_value=1024 * 15,
    )
    mock_is_audio_empty = mocker.patch(
        "speech_recognition.utils.audio_helper.AudioHelper._is_audio_empty",
        return_value=False,
    )
    helper = AudioHelper()
    result = helper.is_file_empty(str(dummy_audio_path))
    assert result is False
    mock_is_audio_empty.assert_called_once()


# --- convert_audio_to_wav tests ---
def test_convert_audio_to_wav_success(mocker, dummy_audio_path, dummy_wav_path):
    mock_audio = mocker.Mock()
    mocker.patch(
        "speech_recognition.utils.audio_helper.pydub.AudioSegment.from_file",
        return_value=mock_audio,
    )

    helper = AudioHelper()
    helper.convert_audio_to_wav(str(dummy_audio_path), str(dummy_wav_path))
    mock_audio.export.assert_called_once_with(str(dummy_wav_path), format="wav")


def test_convert_audio_to_wav_failure(mocker, dummy_audio_path, dummy_wav_path):
    mocker.patch(
        "speech_recognition.utils.audio_helper.pydub.AudioSegment.from_file",
        side_effect=Exception("Something went wrong"),
    )

    helper = AudioHelper()
    with pytest.raises(TranscriptionError):
        helper.convert_audio_to_wav(str(dummy_audio_path), str(dummy_wav_path))


def test_filetype_not_supported(tmp_path, dummy_wav_path):
    path = tmp_path / "file.mb3"

    helper = AudioHelper()
    with pytest.raises(TranscriptionError):
        helper.convert_audio_to_wav(str(path), str(dummy_wav_path))
