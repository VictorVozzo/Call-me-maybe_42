from math import inf
import json
from .llm_sdk.llm_sdk import Small_LLM_Model
from numpy import argmax


class ConstrainedDecoding():
    def __init__(self) -> None:
        self.small_llm = Small_LLM_Model()
        self.create_vocab_file()

    def create_vocab_file(self) -> None:
        vocab_file_path = self.small_llm.get_path_to_vocab_file()
        try:
            with open(vocab_file_path, "r") as file:
                self.vocab_file = json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Vocab file not found: {vocab_file_path}"
            ) from e

    def tokens_numbers(self) -> str:
        ids = []
        print(type(self.vocab_file))
        for key, value in self.vocab_file.items():
            if key.isdigit():
                ids.append(f"{key}-{value}")

        print(ids)

    def constrained_decoding(self, prompt_list) -> str:
        pass
