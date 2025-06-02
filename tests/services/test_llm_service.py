import logging

import pytest

from speech_recognition import LLMService
from speech_recognition.exceptions.llm_processing_error import LLMProcessingError
from speech_recognition.services.llm_service import RequestType


@pytest.fixture(autouse=True)
def disable_logging():
    # Disables logging during tests
    logging.disable(logging.CRITICAL)


@pytest.fixture
def mock_service(mocker):
    mocker.patch(
        "speech_recognition.services.llm_service.AutoModelForCausalLM.from_pretrained",
        return_value=mocker.Mock(),
    )
    mocker.patch(
        "speech_recognition.services.llm_service.AutoTokenizer.from_pretrained",
        return_value=mocker.Mock(),
    )

    return LLMService()


# No test for _generate_output as it only calls external functions and is very annoying to test


def test_load_model(mocker):
    # Mock model and tokenizer so we don't have to load a real one
    mock_model = mocker.patch(
        "speech_recognition.services.llm_service.AutoModelForCausalLM.from_pretrained",
        return_value=mocker.Mock(),
    )
    mock_tokenizer_patch = mocker.patch(
        "speech_recognition.services.llm_service.AutoTokenizer.from_pretrained",
        return_value=mocker.Mock(),
    )

    service = LLMService()

    # Assert that it did something
    assert service is not None
    mock_model.assert_called_once()
    mock_tokenizer_patch.assert_called_once()


def test_generate_json_response(mocker, mock_service):
    # Mock the return of the actual llm generation
    mock_llm = mocker.patch(
        "speech_recognition.services.llm_service.LLMService._LLMService__generate_output",
        return_value='```json{"result": "YES"}```',
    )

    # Call the method
    output = mock_service.generate_json_response("yes", req_type=RequestType.COMMAND)

    # Assert that it did the right things
    assert output == {"result": "YES"}
    mock_llm.assert_called_once()


def test_generate_json_response_bad_request(mocker, mock_service):
    # Mock the output of the llm generation
    mock_llm = mocker.patch(
        "speech_recognition.services.llm_service.LLMService._LLMService__generate_output",
        return_value='```json{"result": "YES"}```',
    )

    # Call it with a bad_request to assert that it fails
    with pytest.raises(LLMProcessingError, match="Invalid request type"):
        mock_service.generate_json_response("bad request", RequestType.BAD_REQUEST)

    # Assert that the actual generation wasn't called
    assert mock_llm.call_count == 0


def test_generate_json_response_output_raises(mocker, mock_service):
    # Mock the llm generation throwing an exception
    mocker.patch(
        "speech_recognition.services.llm_service.LLMService._LLMService__generate_output",
        side_effect=Exception("Something went wrong"),
    )

    # Assert that it caught the exception and threw the custom one
    with pytest.raises(LLMProcessingError, match="Error during processing of prompt"):
        mock_service.generate_json_response("yes", req_type=RequestType.COMMAND)
