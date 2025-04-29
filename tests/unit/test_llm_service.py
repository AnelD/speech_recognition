from speech_recognition import LLMService
from speech_recognition.services.llm_service import RequestType


def test_generate_json_response(mocker):

    mock_model = mocker.patch(
        "speech_recognition.services.llm_service.AutoModelForCausalLM.from_pretrained",
        return_value=mocker.Mock(),
    )
    mock_tokenizer_patch = mocker.patch(
        "speech_recognition.services.llm_service.AutoTokenizer.from_pretrained",
        return_value=mocker.Mock(),
    )

    service = LLMService()

    mock_llm = mocker.patch(
        "speech_recognition.services.llm_service.LLMService._generate_output",
        return_value='```json{"result": "YES"}```',
    )

    # Call the method
    output = service.generate_json_response("yes", req_type=RequestType.COMMAND)

    assert output == '{"result": "YES"}'
    mock_llm.assert_called_once()
    mock_model.assert_called_once()
    mock_tokenizer_patch.assert_called_once()
