import pytest

from speech_recognition import LLMService
from speech_recognition.exceptions.llm_processing_error import LLMProcessingError
from speech_recognition.services.llm_service import RequestType


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
    mock_model = mocker.patch(
        "speech_recognition.services.llm_service.AutoModelForCausalLM.from_pretrained",
        return_value=mocker.Mock(),
    )
    mock_tokenizer_patch = mocker.patch(
        "speech_recognition.services.llm_service.AutoTokenizer.from_pretrained",
        return_value=mocker.Mock(),
    )

    service = LLMService()
    assert service is not None
    mock_model.assert_called_once()
    mock_tokenizer_patch.assert_called_once()


def test_generate_json_response(mocker, mock_service):

    mock_llm = mocker.patch(
        "speech_recognition.services.llm_service.LLMService._generate_output",
        return_value='```json{"result": "YES"}```',
    )

    # Call the method
    output = mock_service.generate_json_response("yes", req_type=RequestType.COMMAND)

    assert output == '{"result": "YES"}'
    mock_llm.assert_called_once()


def test_generate_json_response_bad_request(mocker, mock_service):
    mock_llm = mocker.patch(
        "speech_recognition.services.llm_service.LLMService._generate_output",
        return_value='```json{"result": "YES"}```',
    )

    with pytest.raises(LLMProcessingError, match="Invalid request type"):
        mock_service.generate_json_response("bad request", RequestType.BAD_REQUEST)

    assert mock_llm.call_count == 0


def test_generate_json_response_output_raises(mocker, mock_service):
    mock_llm = mocker.patch(
        "speech_recognition.services.llm_service.LLMService._generate_output",
        side_effect=Exception("Something went wrong"),
    )
    with pytest.raises(LLMProcessingError, match="Error during processing of prompt"):
        mock_service.generate_json_response("yes", req_type=RequestType.COMMAND)
