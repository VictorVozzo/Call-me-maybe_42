from math import inf
from numpy import argmax
import json
from .llm_sdk.llm_sdk import Small_LLM_Model
from .get_data import FunctionsDefinition


class ConstrainedDecoding():
    def __init__(self, functions_json_data: list[FunctionsDefinition], prompt_list: list[str]) -> None:
        self.functions_json_data = functions_json_data
        self.prompt_list = prompt_list
        self.small_llm = Small_LLM_Model()
        self.create_vocab_file()
        self.get_function_names()

    def create_vocab_file(self) -> None:
        vocab_file_path = self.small_llm.get_path_to_vocab_file()
        try:
            with open(vocab_file_path, "r") as file:
                vocab_file = json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Vocab file not found: {vocab_file_path}"
            ) from e

        self.content_id_vocab: dict[int, str] = {}

        for key, value in vocab_file.items():
            self.content_id_vocab[value] = key

    def get_function_names(self) -> None:
        self.functions_names: list[str] = []

        for function in self.functions_json_data:
            self.functions_names.append(function.name)

    def get_parameters(self, function_name: str) -> dict[str | float | int]:
        functions_parameters: dict[str, str] = {}

        for function in self.functions_json_data:
            if function.name is function_name:
                for key, value in function.parameters.items():
                    functions_parameters[key] = value.type

        return functions_parameters

    # receive the prompt and a list of allowed_ids, return the id with the biggest logit
    def find_max_logit_text(self, prompt: str, list_allowed_ids: list[int]) -> int:
        list_logits: dict[int, float] = []
        full_list_logits: list[float] = (
            self.small_llm.get_logits_from_input_ids(
                self.small_llm.encode(prompt).tolist()[0]))

        for id in list_allowed_ids:
            list_logits[id] = full_list_logits[id]

        max_logit = max(list_logits, key=list_logits.get)

        return self.small_llm.decode(max_logit)

    # just testing tokens(not final function)
    def tokens_numbers(self) -> str:
        ids = []

        for function in self.functions_names:
            list_id = self.small_llm.encode(function).tolist()[0]
            for id in list_id:
                print(f"token: '{self.small_llm.decode(id)}' ,ID: '{id}'")
            print(self.get_parameters(function))

            print("\n\n")

    def constrained_decoding(self) -> str:
        partiat_output = ""
