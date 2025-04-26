import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logger_helper import LoggerHelper

log = LoggerHelper(__name__, log_level=logging.DEBUG).get_logger()


class LLMService:
    """Service for loading a language model and generating structured JSON responses."""

    SYSTEM_PROMPT = (
        "You are a data extraction assistant. "
        "Your task is to listen to people's speech transcriptions and extract personal details into a JSON object. "
        "Required fields: firstname, lastname, sex, date_of_birth, phone_number, email_address. "
        "If a field is missing or unclear, set its value to null. "
        "For 'sex', use 'M' for male, 'W' for female, and 'D' for diverse/other. "
        "Replace spoken 'at' or 'dot' appropriately in email addresses. "
        "Return ONLY the raw JSON object, without any commentary, Markdown, or extra text."
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """Initializes the LLMService with the given model name.

        Args:
            model_name (str, optional): Name or path of the model to load.
            Default to "Qwen/Qwen2.5-0.5B-Instruct".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        """Loads the language model and tokenizer onto the appropriate device.

        Returns:
            tuple: The loaded model and tokenizer.
        """
        log.info(f"Loading model: {self.model_name} on device: {self.device}")
        t0 = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        t1 = time.time()
        log.info(f"LLM model loaded in {t1 - t0:.2f} seconds.")
        return model, tokenizer

    def generate_json_response(self, prompt: str) -> str:
        """Generates a structured JSON response based on a natural language prompt.

        Args:
            prompt (str): The user's input text to be processed.

        Returns:
            str: A JSON-formatted string containing the extracted information.
        """
        log.debug(f"Generating response for prompt: {prompt}")

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        t0 = time.time()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            # Make output more deterministic
            do_sample=False,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        t1 = time.time()

        log.info(f"Generated response in {t1 - t0:.2f} seconds.")
        log.debug(f"LLM raw output: {output}")

        output = output.replace("```json", "").replace("```", "").strip()
        return output
