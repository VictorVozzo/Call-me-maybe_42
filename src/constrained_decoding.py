from math import isfinite
import json
from copy import deepcopy
from .llm_sdk.llm_sdk import Small_LLM_Model
from .get_data import FunctionsDefinition, FunctionCalling


class ConstrainedDecoding():
    def __init__(self, functions_json_data: list[FunctionsDefinition],
                 tests_json_data: list[FunctionCalling],
                 prompt_list: list[str]) -> None:
        self.functions_json_data = functions_json_data
        self.tests_json_data = tests_json_data
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

        self.id_content_vocab: dict[int, str] = {}

        for key, value in vocab_file.items():
            self.id_content_vocab[value] = key

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

    # find valid tokens numbers, the last param includes }
    def valid_tokens_numbers(self) -> tuple[list[int], list[int]]:
        valid_tokens: list[int] = []
        valid_tokens_last: list[int] = []

        for id, token in self.id_content_vocab.items():
            try:
                check_token = float(token)
                if not isfinite(check_token):
                    raise
                valid_tokens.append(token)
                valid_tokens_last.append(token)
            except:
                if token == "-" or token == ".":
                    valid_tokens.append(token)
                    valid_tokens_last.append(token)
                elif token == ",":
                    valid_tokens.append(token)
                elif token == "}":
                    valid_tokens_last.append(token)
                elif "," in token and token.count(",") == 1 and token[-1] == ",":
                    try:
                        check_token = float(token[:-1])
                        if not isfinite(check_token):
                            raise
                        valid_tokens.append(token)
                    except:
                        pass
                elif "}" in token and token.count("}") == 1 and token[-1] == "}":
                    try:
                        check_token = float(token[:-1])
                        if not isfinite(check_token):
                            raise
                        valid_tokens_last.append(token)
                    except:
                        pass

        return tuple[valid_tokens, valid_tokens_last]

    # delete functions that are not valid anymore
    def del_invalid_function(self, tokens_writed: list[int],
                             function_ids: dict[str, list[int]]
                             ) -> dict[str, list[int]]:

        function_ids_copy = deepcopy(function_ids)

        for i in range(len(tokens_writed)):
            for key, value in function_ids.items():
                if value[i] != tokens_writed[i]:
                    try:
                        del function_ids_copy[key]
                    except:
                        pass

        print(function_ids_copy)

        function_ids = deepcopy(function_ids_copy)
        return function_ids


    # att functions_json_data to each step of the prompts (NEED TO TEST)
    def del_functions_json_data(function_ids: dict[str, list[int]],
                                functions_json_data_prompt: list[
                                    FunctionsDefinition]) -> None:

        functions_json_data_copy = deepcopy(functions_json_data_prompt)

        for function in functions_json_data_copy:
            if function.name not in function_ids:
                for fnc in functions_json_data_prompt:
                    if fnc.name == function.name:
                        functions_json_data_prompt.remove(fnc)

    def valid_tokens_functions(self) -> dict[str, list[int]]:
        functions_ids: dict[str, list[int]] = {}

        for function in self.functions_names:
            functions_ids[function] = (self.small_llm.encode(
                function).tolist()[0])

        return functions_ids

    #write the functions with LLM
    def write_function() -> str:
        pass

    #write the parameters with LLM
    def write_parameters() -> str:
        pass

    def constrained_decoding(self) -> str:
        final_output = "[\n"
        prompts_output: list[str] = []

        print(self.functions_json_data)
        index = 0
        # for prompt in self.tests_json_data:

        #     partial_prompt = self.prompt_list[index]
        #     prompt_output = f'\n  {{\n    "prompt": {prompt.prompt},\n    "name": '
        #     # write the function name:
        #     # llm choose the function
        #     prompt_output += f',\n    "parameters": '
        #     # write the parameters:
        #     # llm choose the parameters
        #     prompt_output += f'\n  }}'
        #     # print(prompt_output)
        #     index += 1
        #     print(partial_prompt + "\n" + prompt_output)
        #     print("\n\n")

        final_output += "\n]"
        return final_output
