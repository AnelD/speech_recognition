import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from LoggerHelper import LoggerHelper

log = LoggerHelper(__name__, log_level=logging.DEBUG).get_logger()


class LLMService:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
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
        log.debug(f"Generating response for prompt: {prompt}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an information extraction assistant. Your job is to take sentences "
                    "said by some people and to format them into a json format. "
                    "Don't put formatting quotes. You will usually receive a firstname, lastname, sex, "
                    "date of birth, phone number, and email address. "
                    "Make all of those separate fields in the JSON and try to replace "
                    "a spoken 'at' with '@' in their email address. For sex return M (male), W (female), or D (other). "
                    "If a field is missing, set it to null. Return as plain text."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)

        t0 = time.time()
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
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
