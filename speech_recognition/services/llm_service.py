import time
from enum import Enum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from speech_recognition import config
from speech_recognition.exceptions.llm_processing_error import LLMProcessingError
from speech_recognition.utils.logger_helper import LoggerHelper

log = LoggerHelper(__name__).get_logger()


class RequestType(Enum):
    BAD_REQUEST = 1
    PERSON_DATA = 2
    COMMAND = 3


class LLMService:
    """Service for loading a language model and generating structured JSON responses."""

    _PERSON_DATA_PROMPT = """
        You are a data extraction assistant. 
        Your task is to listen to people's speech transcriptions and extract personal details into a JSON object. 
        Required fields: firstname, lastname, sex, date_of_birth, phone_number, email_address. 
        If a field is missing or unclear, set its value to null. 
        For 'sex', use 'M' for male, 'W' for female, and 'D' for diverse/other. 
        Replace spoken 'at' or 'dot' appropriately in email addresses. 
        Return ONLY the raw JSON object, without any commentary, Markdown, or extra text.
    """

    _COMMAND_PROMPT = """
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
        """Initializes the LLMService with configuration from config.py."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config.LLM_MODEL_NAME
        self.model, self.tokenizer = self._load_model()

    def generate_json_response(self, prompt: str, req_type: RequestType) -> str:
        """Generates a structured JSON response based on a natural language prompt.

        Args:
            prompt (str): The user's input text to be processed.
            req_type (RequestType): The type of request to generate a response for.

        Returns:
            str: A JSON-formatted string containing the extracted information.
        """
        log.debug(f"Generating response for prompt: {prompt}")

        match req_type:
            case RequestType.PERSON_DATA:
                system_prompt = self._PERSON_DATA_PROMPT
            case RequestType.COMMAND:
                system_prompt = self._COMMAND_PROMPT
            case _:
                log.error(f"Invalid request type: {req_type}")
                raise LLMProcessingError(f"Invalid request type: {req_type}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        t0 = time.time()
        try:
            output = self._generate_output(messages)
        except Exception as e:
            log.error(f"Error generating response: {e}")
            raise LLMProcessingError(f"Error during processing of prompt: {messages}")
        t1 = time.time()

        log.info(f"Generated response in {t1 - t0:.2f} seconds.")
        log.debug(f"LLM raw output: {output}")

        output = output.replace("```json", "").replace("```", "").strip()
        return output

    def _load_model(self) -> (AutoModelForCausalLM, AutoTokenizer):
        """Loads the language model and tokenizer onto the appropriate device.

        Returns:
            tuple: The loaded model and tokenizer.
        """
        log.info(f"Loading model: {self.model_name} on device: {self.device}")
        t0 = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        t1 = time.time()
        log.info(f"LLM model loaded in {t1 - t0:.2f} seconds.")
        return model, tokenizer

    def _generate_output(self, messages):
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            # Make output more deterministic
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
