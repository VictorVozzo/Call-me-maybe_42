from math import isfinite, inf
import json
from copy import deepcopy
from .llm_sdk.llm_sdk import Small_LLM_Model
from .get_data import FunctionsDefinition, FunctionCalling
from .prompt import get_prompt


class ConstrainedDecoding():
    def __init__(self, functions_json_data: list[FunctionsDefinition],
                 tests_json_data: list[FunctionCalling]) -> None:
        self.functions_json_data = functions_json_data
        self.tests_json_data = tests_json_data
        self.small_llm = Small_LLM_Model()
        self.create_vocab_file()
        self.get_function_names()
        self.valid_tokens_numbers: tuple[
            list[int], list[int], list[int]] = self.get_valid_tokens_numbers()
        self.valid_tokens_boolean: tuple[
            list[int], list[int]] = self.get_valid_tokens_boolean()
        self.valid_tokens_string: list[int] = self.get_valid_tokens_string()

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
    def find_max_logit_id(self, prompt: str, list_allowed_ids: set[int]) -> int:
        list_logits: dict[int, float] = {}
        full_list_logits: list[float] = (
            self.small_llm.get_logits_from_input_ids(
                self.small_llm.encode(prompt).tolist()[0]))

        for id in list_allowed_ids:
            list_logits[id] = full_list_logits[id]

        max_logit = max(list_logits, key=list_logits.get)

        return max_logit

    # find valid tokens numbers, the last param includes '}' and the others ','
    def get_valid_tokens_numbers(self) -> tuple[list[int], list[int], list[int]]:
        valid_minus_token: list[int] = []
        valid_tokens: list[int] = []
        valid_tokens_last: list[int] = []

        for id, token in self.id_content_vocab.items():
            try:
                check_token = float(token)
                if not isfinite(check_token):
                    raise
                valid_tokens.append(id)
                valid_tokens_last.append(id)
            except:
                if token == "-":
                    valid_minus_token.append(id)
                elif token == ".":
                    valid_tokens.append(id)
                    valid_tokens_last.append(id)
                elif token == ",":
                    valid_tokens.append(id)
                elif token == "}":
                    valid_tokens_last.append(id)
                elif "," in token and token.count(",") == 1 and token[-1] == ",":
                    try:
                        check_token = float(token[:-1])
                        if not isfinite(check_token):
                            raise
                        valid_tokens.append(id)
                    except:
                        pass
                elif "}" in token and token.count("}") == 1 and token[-1] == "}":
                    try:
                        check_token = float(token[:-1])
                        if not isfinite(check_token):
                            raise
                        valid_tokens_last.append(id)
                    except:
                        pass

        return (valid_tokens, valid_tokens_last, valid_minus_token)

    # valid boolean tokens(True or False), gonna check if the llm ended writing by the len of the list of ids - same idea as the functions
    def get_valid_tokens_boolean(self) -> tuple[list[int], list[int]]:
        valid_tokens_true = self.small_llm.encode("true").tolist()[0]
        valid_tokens_false = self.small_llm.encode("false").tolist()[0]

        return (valid_tokens_true, valid_tokens_false)

    def get_valid_tokens_string(self) -> list[int]:
        valid_tokens = []

        for id, token in self.id_content_vocab.items():

            if token.count('"') == 1 and token[-1] == '"':
                valid_tokens.append(id)
            elif token.count('"') == 0:
                valid_tokens.append(id)

        return valid_tokens

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

        new_function_ids = deepcopy(function_ids_copy)
        return new_function_ids

    # att functions_json_data to each step of the prompts

    def del_functions_json_data(self, function_ids: dict[str, list[int]],
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

    # write the functions with LLM (return the prompt that will use in the write parameters)
    def write_function(self, partial_prompt: str,
                       prompt_output: str,
                       function_ids: dict[str, list[int]],
                       functions_json_data_prompt: list[
                           FunctionsDefinition],
                       user_prompt: str) -> tuple[str, str]:

        new_function_ids = deepcopy(function_ids)
        biggest_function = max(len(ids) for ids in new_function_ids.values())
        writed_functions_id = []

        i = 0
        while i < biggest_function:
            allowed_ids = set()
            for id in new_function_ids.values():
                allowed_ids.add(id[i])
            if len(new_function_ids) > 1:
                biggest_logit_id = self.find_max_logit_id(
                    partial_prompt + prompt_output, allowed_ids)
            else:
                for bli in allowed_ids:
                    biggest_logit_id = bli
            prompt_output += self.small_llm.decode(biggest_logit_id)
            writed_functions_id.append(biggest_logit_id)
            if len(new_function_ids) > 1:
                new_function_ids = self.del_invalid_function(
                    writed_functions_id, new_function_ids)
                self.del_functions_json_data(
                    new_function_ids, functions_json_data_prompt)
                partial_prompt = get_prompt(
                    functions_json_data_prompt, user_prompt)
                biggest_function = max(len(ids)
                                    for ids in new_function_ids.values())
            i += 1

        return (partial_prompt, prompt_output)

    # write the parameters with LLM
    def write_parameters(self, partial_prompt: str,
                       prompt_output: str,
                       functions_json_data_prompt: list[
                           FunctionsDefinition]) -> str:
        
        parameters = functions_json_data_prompt[0].parameters

        valid_tokens_numbers, valid_tokens_numbers_last, valid_minus_token = self.valid_tokens_numbers
        
        valid_tokens_bool_true, valid_tokens_bool_false = self.valid_tokens_boolean
        valid_tokens_bool = {"True": valid_tokens_bool_true, "False": valid_tokens_bool_false}
        biggest_bool = max(len(ids) for ids in valid_tokens_bool.values())

        for parameter, value in parameters.items():
            if parameter != next(iter(parameters)):
                prompt_output += " "
            prompt_output += f"\"{parameter}\": "
            if value.type == "number":

                #id of the dot "." , need to count dots writed so far, if there is one, exclude from the valid_tokens list
                dot_id = self.small_llm.encode(".").tolist()[0]
                
                if parameter != next(reversed(parameters)):
                    if parameter == next(iter(parameters)):
                        valid_tokens = valid_tokens_numbers + valid_minus_token
                    else:
                        valid_tokens = valid_tokens_numbers
                    while prompt_output[-1] != ",":
                        if parameter == next(iter(parameters)):
                            if prompt_output.split("{")[1].count(".") >= 1:
                                valid_tokens.remove(dot_id)
                        else:
                            if prompt_output.split(",")[1].count(".") >= 1:
                                valid_tokens.remove(dot_id)
                        
                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output, valid_tokens)
                        prompt_output += self.small_llm.decode(biggest_logit_id)
                else:
                    if parameter == next(iter(parameters)):
                        valid_tokens = valid_tokens_numbers_last + valid_minus_token
                    else:
                        valid_tokens = valid_tokens_numbers_last
                    while prompt_output[-1] != "}":
                        if parameter == next(iter(parameters)):
                            if prompt_output.split("{")[1].count(".") >= 1:
                                valid_tokens.remove(dot_id)
                        else:
                            if prompt_output.split(",")[1].count(".") >= 1:
                                valid_tokens.remove(dot_id)
                        
                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output, valid_tokens)
                        prompt_output += self.small_llm.decode(biggest_logit_id)
            elif value.type == "boolean":
                biggest_logit_id = self.find_max_logit_id(
                    partial_prompt + prompt_output, {valid_tokens_bool_true[0], valid_tokens_bool_false[0]})
                if biggest_logit_id == valid_tokens_bool_true[0]:
                    for i in range(len(valid_tokens_bool_true)):
                        prompt_output += self.small_llm.decode(valid_tokens_bool_true[i])
                else:
                    for j in range(len(valid_tokens_bool_false)):
                        prompt_output += self.small_llm.decode(valid_tokens_bool_false[j])
                if parameter != next(reversed(parameters)):
                    prompt_output += ","
                else:
                    prompt_output += "}"
            else:
                prompt_output += '"'

                flag = True
                while prompt_output[-1] != '"' or flag is True:
                    biggest_logit_id = self.find_max_logit_id(
                        partial_prompt + prompt_output, self.valid_tokens_string)
                    prompt_output += self.small_llm.decode(biggest_logit_id)
                    flag = False
                    print("\n\n")
                    print(partial_prompt + prompt_output)
                
                if parameter != next(reversed(parameters)):
                    prompt_output += ","
                else:
                    prompt_output += "}"

        return prompt_output

    def constrained_decoding(self) -> str:
        final_output = "[\n"
        prompts_output: list[str] = []
        function_ids = self.valid_tokens_functions()

        print("\n\n")
        for prompt in self.tests_json_data:
            functions_json_data_prompt = deepcopy(self.functions_json_data)
            partial_prompt = get_prompt(functions_json_data_prompt, prompt.prompt)
            prompt_output = f'\n  {{\n    "prompt": \"{prompt.prompt}\",\n    "name": \"'
            # write the function name (llm choose the function):
            partial_prompt, prompt_output = self.write_function(partial_prompt,
                                                                prompt_output, function_ids,
                                                                functions_json_data_prompt,
                                                                prompt.prompt)
            prompt_output += f'\",\n    "parameters": {{'
            # write the parameters(llm choose the parameters):
            prompt_output = self.write_parameters(partial_prompt, 
                                                  prompt_output, 
                                                  functions_json_data_prompt)
            prompt_output += f'\n  }}'
            print(partial_prompt + prompt_output)
            print("\n\n\n\n")

        final_output += "\n]"
        return final_output
