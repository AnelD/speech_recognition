import json
import time
from enum import Enum

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)

from speech_recognition import config
from speech_recognition.exceptions.llm_processing_error import LLMProcessingError
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class RequestType(Enum):
    BAD_REQUEST = 1
    PERSON_DATA = 2
    COMMAND = 3


class LLMService:
    """Service for loading a language model and generating structured JSON responses.

    This service leverages a Hugging Face causal language model to process transcription
    prompts and return structured outputs in JSON format. It supports extracting personal
    data or interpreting commands with yes/no logic.

    Attributes:
        __device (torch.device): The device (CPU/GPU) on which the model will run.
        __model_name (str): Name or path of the pretrained model from configuration.
        __model (PreTrainedModel): Loaded causal language model for inference.
        __tokenizer (PreTrainedTokenizerFast): Tokenizer used to encode/decode prompts.
        __PERSON_DATA_PROMPT (str): Prompt instructing the model to extract person-related fields.
        __COMMAND_PROMPT (str): Prompt for interpreting input as a binary command (yes/no).
    """

    __PERSON_DATA_PROMPT = """
        You are a data extraction assistant. 
        Your task is to listen to people's speech transcriptions and extract personal details into a JSON object. 
        Required fields: firstname, lastname, sex, date_of_birth, phone_number, email_address. 
        If a field is missing or unclear, set its value to null. 
        For 'sex', use 'M' for male, 'W' for female, and 'D' for diverse/other. 
        Replace spoken 'at' or 'dot' appropriately in email addresses. 
        Return ONLY the raw JSON object, without any commentary, Markdown, or extra text.
    """

    __COMMAND_PROMPT = """
        You are a data extraction assistant. 
        Your task is to listen to people's speech transcriptions and Classify the input text strictly:
        Be very strict. Only classify as yes or no if it is clear.
        If the text expresses a YES (agreement, acceptance, affirmation), output { "result": "YES" }
        If the text expresses a NO (refusal, rejection, declination), output { "result": "NO" }
        If the text does not express yes or no (e.g., it is off-topic, random, unrelated, 
        like just saying a name or making a comment), output { "error": "Input is not a yes or no." }
        Return ONLY the raw JSON object, without any commentary, Markdown, or extra text.
    """

    def __init__(self) -> None:
        """Initializes the LLMService with a language model and tokenizer.

        Loads the model and tokenizer defined in the configuration onto the
        appropriate device (GPU if available, otherwise CPU).
        """

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model_name = config.LLM_MODEL_NAME
        self.__model, self.__tokenizer = self.__load_model()

    def generate_json_response(self, prompt: str, req_type: RequestType) -> dict:
        """Generates a structured JSON response from a given prompt and request type.

        Based on the request type, selects the appropriate system prompt and sends
        the prompt to the language model. Parses the JSON-formatted response and
        returns it as a dictionary.

        Args:
            prompt (str): The input text to process (e.g., transcribed user speech).
            req_type (RequestType): Type of request (PERSON_DATA or COMMAND).

        Returns:
            dict: The structured information extracted from the model's output.

        Raises:
            LLMProcessingError: If the request type is invalid or model inference fails.
        """
        log.debug(f"Generating response for prompt: {prompt}")

        match req_type:
            case RequestType.PERSON_DATA:
                system_prompt = self.__PERSON_DATA_PROMPT
            case RequestType.COMMAND:
                system_prompt = self.__COMMAND_PROMPT
            case _:
                log.error(f"Invalid request type: {req_type}")
                raise LLMProcessingError(f"Invalid request type: {req_type}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        t0 = time.time()
        try:
            output = self.__generate_output(messages)
        except Exception as e:
            log.error(f"Error generating response: {e}")
            raise LLMProcessingError(f"Error during processing of prompt: {messages}")
        t1 = time.time()

        log.info(f"Generated response in {t1 - t0:.2f} seconds.")
        log.debug(f"LLM raw output: {output}")

        output = output.replace("```json", "").replace("```", "").strip()
        return json.loads(output)

    def __generate_output(self, messages: list[dict[str, str]]) -> str:
        """Generates raw text output from a list of chat-style messages.

        Uses the tokenizer's chat template to format input and generates output
        using the loaded model.

        Args:
            messages (list[dict[str, str]]): A list of chat messages including system
                and user roles for prompt context.

        Returns:
            str: The raw output string generated by the model.
        """
        input_text = self.__tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.__tokenizer([input_text], return_tensors="pt").to(
            self.__model.device
        )

        generated_ids = self.__model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.__tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def __load_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        """Loads the language model and tokenizer.

        Loads the causal language model and tokenizer using Hugging Face's
        `from_pretrained` method, and sets them to the appropriate device.

        Returns:
            tuple: A tuple containing the loaded model and tokenizer.
        """
        log.info(f"Loading model: {self.__model_name} on device: {self.__device}")
        t0 = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            self.__model_name, device_map="auto", torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.__model_name)

        t1 = time.time()
        log.info(f"LLM model loaded in {t1 - t0:.2f} seconds.")
        return model, tokenizer
