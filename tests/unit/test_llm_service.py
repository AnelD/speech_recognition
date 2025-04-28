import torch

from speech_recognition import LLMService
from speech_recognition.services.llm_service import RequestType


def test_generate_json_response(mocker):
    # Patch model
    mock_model = mocker.Mock()
    mock_model.generate.return_value = torch.tensor([[0, 1, 2]])

    # Patch tokenizer
    mock_tokenizer = mocker.Mock()
    mock_tokenizer.apply_chat_template.return_value = "input text"
    mock_tokenizer.return_value = mocker.Mock()
    mock_tokenizer.batch_decode.return_value = [
        '{"result": "YES"}'
    ]

    mock_llm = mocker.patch(
        "speech_recognition.services.llm_service.AutoModelForCausalLM.from_pretrained",
        return_value=mock_model,
    )
    mock_tokenizer_patch = mocker.patch(
        "speech_recognition.services.llm_service.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )

    service = LLMService()

    # Call the method
    output = service.generate_json_response("hello", req_type=RequestType.COMMAND)

    assert output == '{"result": "YES"}'
    mock_llm.assert_called_once()
    mock_tokenizer_patch.assert_called_once()
    mock_tokenizer.batch_decode.assert_called_once()
