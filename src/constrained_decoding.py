# *************************************************************************** #
#                                                                             #
#                                                         :::      ::::::::   #
#   constrained_decoding.py                             :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#   By: vivozzo- <vivozzo-@student.42porto.com>     +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#   Created: 2026/11/16 17:08:16 by vivozzo-           #+#    #+#             #
#   Updated: 2026/11/16 18:49:45 by vivozzo-          ###   ########.fr       #
#                                                                             #
# *************************************************************************** #


from math import isfinite
import json
from copy import deepcopy
from time import time
import os
from .llm_sdk.llm_sdk import Small_LLM_Model
from .get_data import FunctionsDefinition, FunctionCalling
from .prompt import get_prompt


class ConstrainedDecoding():
    """Generate schema-compliant function calls with constrained decoding.

    At every generation step the LLM logits are restricted to the tokens
    that keep the output valid (a known function name, a number, a boolean
    or a string), so the produced JSON is always parseable and follows the
    schema of the chosen function.

    Attributes:
        functions_json_data: Available function definitions.
        tests_json_data: Prompts to convert into function calls.
        output_path: Path of the JSON file that will be written.
        small_llm: Wrapper used to talk to the LLM.
        id_content_vocab: Mapping of token id to token text.
        functions_names: Names of all available functions.
        valid_tokens_numbers: Four lists of token ids: tokens allowed
            before a ``,``, tokens allowed before a ``}``, negative
            number tokens and tokens that contain a decimal point.
        valid_tokens_boolean: Token ids of ``true`` and ``false``.
        valid_tokens_string: Token ids allowed inside a string, and the
            ones allowed for the last string parameter.
        terminal_output: Text accumulated for the live terminal display.
    """

    def __init__(self, functions_json_data: list[FunctionsDefinition],
                 tests_json_data: list[FunctionCalling],
                 output_path: str) -> None:
        """Initialize the model, the vocabulary and the valid token sets.

        Args:
            functions_json_data: Available function definitions.
            tests_json_data: Prompts to convert into function calls.
            output_path: Path of the output JSON file.

        Raises:
            FileNotFoundError: If the vocabulary file cannot be found.
        """

        self.functions_json_data = functions_json_data
        self.tests_json_data = tests_json_data
        self.output_path = output_path
        self.small_llm = Small_LLM_Model()
        self.create_vocab_file()
        self.get_function_names()
        self.valid_tokens_numbers: tuple[
            list[int], list[int], list[int],
            list[int]] = self.get_valid_tokens_numbers()
        self.valid_tokens_boolean: tuple[
            list[int], list[int]] = self.get_valid_tokens_boolean()
        self.valid_tokens_string: tuple[
            list[int], list[int]] = self.get_valid_tokens_string()
        self.terminal_output = ""

    def create_vocab_file(self) -> None:
        """Load the vocabulary file and build the id to token mapping.

        Fills ``self.id_content_vocab`` by inverting the token to id mapping
        stored in the vocabulary JSON file.

        Raises:
            FileNotFoundError: If the vocabulary file does not exist.
        """

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
        """Store the available function names in ``functions_names``."""

        self.functions_names: list[str] = []

        for function in self.functions_json_data:
            self.functions_names.append(function.name)

    def find_max_logit_id(self, prompt: str,
                          list_allowed_ids: set[int]) -> int:
        """Return the allowed token with the highest logit.

        Args:
            prompt: Text given to the LLM (prompt plus text generated so far).
            list_allowed_ids: Token ids that may be selected.

        Returns:
            The id, among the allowed ones, with the biggest logit.
        """

        full_list_logits: list[float] = (
            self.small_llm.get_logits_from_input_ids(
                self.small_llm.encode(prompt).tolist()[0]))

        return max(
            list_allowed_ids,
            key=lambda token_id: full_list_logits[token_id]
        )

    def get_valid_tokens_numbers(self) -> tuple[list[int],
                                                list[int], list[int],
                                                list[int]]:
        valid_minus_token: list[int] = []
        valid_tokens: list[int] = []
        valid_tokens_last: list[int] = []
        point_tokens: list[int] = []
        """Classify vocabulary tokens that can be part of a JSON number.

        Tokens are checked together with an optional leading ``-`` and an
        optional trailing ``,`` or ``}`` (which end the current parameter).

        Returns:
            A tuple of four lists of token ids:
            ``(valid_tokens, valid_tokens_last, valid_minus_token,
            point_tokens)``. The first one is used for parameters followed
            by ``,``, the second for the last parameter (followed by ``}``),
            the third holds negative number tokens and the last one holds
            tokens that contain a decimal point.
        """

        for id, token in self.id_content_vocab.items():
            flag_minus = False
            flag_middle = False
            flag_last = False

            try:
                check_token = float(token)
                if not isfinite(check_token):
                    raise
                if token.count(".") == 1:
                    point_tokens.append(id)
                if check_token < 0:
                    valid_minus_token.append(id)
                else:
                    valid_tokens.append(id)
                    valid_tokens_last.append(id)
            except Exception:
                if "-" in token and token.count("-") == 1 and token[0] == "-":
                    if token == "-":
                        valid_minus_token.append(id)
                        continue
                    token = token[1:]
                    flag_minus = True

                if "," in token and token.count(",") == 1 and token[-1] == ",":
                    if token == "," and not flag_minus:
                        valid_tokens.append(id)
                        continue
                    token = token[:-1]
                    flag_middle = True
                elif ("}" in token and token.count("}") == 1 and
                      token[-1] == "}"):
                    if token == "}" and not flag_minus:
                        valid_tokens_last.append(id)
                        continue
                    token = token[:-1]
                    flag_last = True

                if token.count(".") == 1:
                    if (token == "." and not flag_minus and
                            not flag_middle and not flag_last):
                        valid_tokens.append(id)
                        valid_tokens_last.append(id)
                        point_tokens.append(id)
                        continue
                    token_split = token.split(".")
                    try:
                        check_token = float(token_split[0])
                        if not isfinite(check_token):
                            raise
                        check_token = float(token_split[1])
                        if not isfinite(check_token):
                            raise
                        if flag_minus:
                            valid_minus_token.append(id)
                        valid_tokens.append(id)
                        valid_tokens_last.append(id)
                        point_tokens.append(id)
                        continue
                    except Exception:
                        pass

                try:
                    check_token = float(token)
                    if not isfinite(check_token):
                        raise
                    if flag_minus:
                        valid_minus_token.append(id)
                    if flag_middle:
                        valid_tokens.append(id)
                    if flag_last:
                        valid_tokens_last.append(id)
                except Exception:
                    pass

        return (valid_tokens, valid_tokens_last,
                valid_minus_token, point_tokens)

    def get_valid_tokens_boolean(self) -> tuple[list[int], list[int]]:
        """Encode the boolean literals ``true`` and ``false``.

        Returns:
            A tuple ``(true_ids, false_ids)`` with the token ids of each
            literal.
        """

        valid_tokens_true = self.small_llm.encode("true").tolist()[0]
        valid_tokens_false = self.small_llm.encode("false").tolist()[0]

        return (valid_tokens_true, valid_tokens_false)

    def get_valid_tokens_string(self) -> tuple[list[int], list[int]]:
        """Classify vocabulary tokens that can be part of a JSON string.

        Tokens without a double quote are always valid. Tokens with a single
        closing quote are valid only if followed by nothing or ``,`` (middle
        parameters) or ``}`` (last parameter).

        Returns:
            A tuple ``(valid_tokens, valid_tokens_last)`` with the token ids
            for middle parameters and for the last parameter.
        """

        valid_tokens = []
        valid_tokens_last = []

        for id, token in self.id_content_vocab.items():
            if (token.count('"') == 1 and
                (token[-1] == '"' or
                 (token[-2] == '"' and token[-1] == ","))):
                valid_tokens.append(id)
            elif (token.count('"') == 1 and
                  token[-2] == '"' and token[-1] == "}"):
                valid_tokens_last.append(id)
            elif token.count('"') == 0:
                valid_tokens.append(id)
                valid_tokens_last.append(id)

        return (valid_tokens, valid_tokens_last)

    def del_invalid_function(self, tokens_writed: list[int],
                             function_ids: dict[str, list[int]]
                             ) -> dict[str, list[int]]:
        """Remove the functions that no longer match the tokens written so far.

        Args:
            tokens_writed: Token ids already generated for the function name.
            function_ids: Mapping of function name to its token ids.

        Returns:
            A new mapping with only the functions whose first tokens are
            equal to ``tokens_writed``.
        """

        function_ids_copy = deepcopy(function_ids)

        for i in range(len(tokens_writed)):
            for key, value in function_ids.items():
                if value[i] != tokens_writed[i]:
                    function_ids_copy.pop(key, None)

        new_function_ids = deepcopy(function_ids_copy)
        return new_function_ids

    def del_functions_json_data(self, function_ids: dict[str, list[int]],
                                functions_json_data_prompt: list[
                                    FunctionsDefinition]) -> None:
        """Remove from the list the functions that are no longer candidates.

        The list is modified in place so the prompt only describes the
        functions that can still be chosen.

        Args:
            function_ids: Mapping of the remaining candidate function names.
            functions_json_data_prompt: Function definitions of the current
                prompt (modified in place).
        """

        functions_json_data_copy = deepcopy(functions_json_data_prompt)

        for function in functions_json_data_copy:
            if function.name not in function_ids:
                for fnc in functions_json_data_prompt:
                    if fnc.name == function.name:
                        functions_json_data_prompt.remove(fnc)

    def valid_tokens_functions(self) -> dict[str, list[int]]:
        """Encode every function name into token ids.

        Returns:
            A mapping of function name to its list of token ids.
        """

        functions_ids: dict[str, list[int]] = {}

        for function in self.functions_names:
            functions_ids[function] = (self.small_llm.encode(
                function).tolist()[0])

        return functions_ids

    def write_function(self, partial_prompt: str,
                       prompt_output: str,
                       function_ids: dict[str, list[int]],
                       functions_json_data_prompt: list[
                           FunctionsDefinition],
                       user_prompt: str) -> tuple[str, str]:
        """Let the LLM choose the function name, token by token.

        At each step only tokens that continue one of the remaining
        function names are allowed. When a single candidate remains, its
        remaining tokens are written without consulting the LLM.

        Args:
            partial_prompt: Prompt sent to the LLM.
            prompt_output: JSON text generated so far.
            function_ids: Mapping of function name to its token ids.
            functions_json_data_prompt: Function definitions of the current
                prompt (reduced in place while candidates are discarded).
            user_prompt: Original user request, used to rebuild the prompt.

        Returns:
            A tuple ``(partial_prompt, prompt_output)`` with the updated
            prompt and the JSON text including the chosen function name.
        """

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

            choosed_token = self.small_llm.decode([biggest_logit_id])
            prompt_output += choosed_token
            print("\n" * 50 + self.terminal_output + prompt_output)
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

    def write_parameters(self, partial_prompt: str,
                         prompt_output: str,
                         functions_json_data_prompt: list[
                             FunctionsDefinition]) -> str:
        """Let the LLM write the parameters of the chosen function.

        Numbers, booleans and strings are generated with different sets of
        allowed tokens, so every value respects the type declared in the
        function definition. Numbers that are not integers always get a
        decimal part (e.g. ``2`` becomes ``2.0``).

        Args:
            partial_prompt: Prompt sent to the LLM.
            prompt_output: JSON text generated so far.
            functions_json_data_prompt: Function definitions left after the
                function choice; the first item is the chosen function.

        Returns:
            The JSON text with all parameters written, ending with ``}``.
        """

        parameters = functions_json_data_prompt[0].parameters

        (valid_tokens_numbers,
         valid_tokens_numbers_last,
         valid_minus_token,
         point_tokens) = self.valid_tokens_numbers
        (valid_tokens_string,
         valid_tokens_string_last) = self.valid_tokens_string

        (valid_tokens_bool_true,
         valid_tokens_bool_false) = self.valid_tokens_boolean

        for parameter, value in parameters.items():
            if parameter != next(iter(parameters)):
                prompt_output += " "
            prompt_output += f"\"{parameter}\": "
            if (value.type == "number" or value.type == "integer"
                    or value.type == "float"):

                if parameter != next(reversed(parameters)):
                    if parameter == next(iter(parameters)):
                        valid_tokens = (valid_tokens_numbers.copy() +
                                        valid_minus_token.copy())
                    else:
                        valid_tokens = valid_tokens_numbers.copy()
                    removed_points = False
                    parameter_output = ""
                    for _ in range(32):  # 32 tokens max
                        if (not removed_points and
                            (value.type == "integer" or
                             prompt_output.split(":")[-1].count(".") >= 1)):
                            for token in point_tokens:
                                if token in valid_tokens:
                                    valid_tokens.remove(token)
                            removed_points = True

                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output, set(valid_tokens))
                        choosed_token = self.small_llm.decode(
                            [biggest_logit_id])
                        prompt_output += choosed_token
                        parameter_output += choosed_token
                        print("\n" * 50 + self.terminal_output + prompt_output)
                        if prompt_output[-1] == ",":
                            if (value.type != "integer" and
                                    "." not in parameter_output):
                                prompt_output = prompt_output[:-1]
                                prompt_output += ".0,"
                            break
                else:
                    if parameter == next(iter(parameters)):
                        valid_tokens = (valid_tokens_numbers_last.copy() +
                                        valid_minus_token.copy())
                    else:
                        valid_tokens = valid_tokens_numbers_last.copy()
                    removed_points = False
                    parameter_output = ""
                    for _ in range(32):
                        if (not removed_points and
                            (value.type == "integer" or
                             prompt_output.split(":")[-1].count(".") >= 1)):
                            for token in point_tokens:
                                if token in valid_tokens:
                                    valid_tokens.remove(token)
                            removed_points = True

                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output, set(valid_tokens))
                        choosed_token = self.small_llm.decode(
                            [biggest_logit_id])
                        prompt_output += choosed_token
                        parameter_output += choosed_token
                        print("\n" * 50 + self.terminal_output + prompt_output)
                        if prompt_output[-1] == "}":
                            if (value.type != "integer" and
                                    "." not in parameter_output):
                                prompt_output = prompt_output[:-1]
                                prompt_output += ".0}"
                            break

            elif value.type == "boolean":
                biggest_logit_id = self.find_max_logit_id(
                    partial_prompt + prompt_output, {
                        valid_tokens_bool_true[0], valid_tokens_bool_false[0]})
                if biggest_logit_id == valid_tokens_bool_true[0]:
                    for i in range(len(valid_tokens_bool_true)):
                        prompt_output += self.small_llm.decode(
                            [valid_tokens_bool_true[i]])

                        print("\n" * 50 + self.terminal_output + prompt_output)
                else:
                    for j in range(len(valid_tokens_bool_false)):
                        prompt_output += self.small_llm.decode(
                            [valid_tokens_bool_false[j]])

                        print("\n" * 50 + self.terminal_output + prompt_output)
                if parameter != next(reversed(parameters)):
                    prompt_output += ","
                else:
                    prompt_output += "}"
                print("\n" * 50 + self.terminal_output + prompt_output)
            else:
                prompt_output += '"'
                quotation_marks = {
                    "“": "”",   # U+201C -> U+201D
                    "„": "“",   # U+201E -> U+201C
                    "«": "»",   # U+00AB -> U+00BB
                    "❝": "❞",   # U+275D -> U+275E
                    "「": "」",  # U+300C -> U+300D
                    "『": "』",  # U+300E -> U+300F
                    "〝": "〞",  # U+301D -> U+301E
                }
                if parameter != next(reversed(parameters)):
                    parameter_output = ""
                    valid_tokens = valid_tokens_string.copy()

                    for _ in range(64):  # 64 tokens max
                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output,
                            set(valid_tokens))
                        choosed_token = self.small_llm.decode(
                            [biggest_logit_id])

                        for left_quote, right_quote in quotation_marks.items():
                            if right_quote in choosed_token:
                                if left_quote not in parameter_output:
                                    choosed_token = '"'

                        prompt_output += choosed_token
                        parameter_output += choosed_token
                        print("\n" * 50 + self.terminal_output + prompt_output)
                        if (prompt_output[-1] == '"' or
                            (prompt_output[-1] == "," and
                             prompt_output[-2] == '"')):
                            break

                    if prompt_output[-1] != ",":
                        prompt_output += ","
                        print("\n" * 50 + self.terminal_output + prompt_output)
                else:
                    parameter_output = ""
                    valid_tokens = valid_tokens_string_last.copy()

                    for _ in range(64):
                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output,
                            set(valid_tokens))
                        choosed_token = self.small_llm.decode(
                            [biggest_logit_id])
                        print(f"Choosed token: {choosed_token}")
                        for left_quote, right_quote in quotation_marks.items():
                            # print("entrou aqui")
                            if right_quote in choosed_token:
                                print("entrou aqui")
                                if left_quote not in parameter_output:
                                    choosed_token = '"'

                        prompt_output += choosed_token
                        parameter_output += choosed_token
                        print("\n" * 50 + self.terminal_output + prompt_output)

                        if (prompt_output[-1] == '"' or
                            (prompt_output[-1] == "}" and
                             prompt_output[-2] == '"')):
                            break

                    if prompt_output[-1] != "}":
                        prompt_output += "}"
                        print("\n" * 50 + self.terminal_output + prompt_output)

        return prompt_output

    def fix_invalid_json_escapes(self, json_string: str) -> str:
        """Escape the backslashes that make a JSON string invalid.

        Walks through the text and, inside strings, doubles every backslash
        that is not part of a valid JSON escape sequence.

        Args:
            json_string: JSON text that may contain invalid escapes.

        Returns:
            The JSON text with the invalid escapes fixed.
        """

        result = []
        valid_escapes = '"\\/bfnrt'
        hex_digits = "0123456789abcdefABCDEF"

        inside_string = False
        i = 0

        while i < len(json_string):
            char = json_string[i]

            if not inside_string:
                result.append(char)

                if char == '"':
                    inside_string = True

                i += 1
                continue

            if char == '"':
                result.append(char)
                inside_string = False
                i += 1
                continue

            if char == "\\":
                # Backslash at the end of the input
                if i + 1 >= len(json_string):
                    result.append("\\\\")
                    i += 1
                    continue

                next_char = json_string[i + 1]

                # Valid standard JSON escape
                if next_char in valid_escapes:
                    result.append(char + next_char)
                    i += 2
                    continue

                # Valid Unicode escape: \uXXXX
                if next_char == "u":
                    unicode_digits = json_string[i + 2:i + 6]

                    if (
                        len(unicode_digits) == 4
                        and all(c in hex_digits for c in unicode_digits)
                    ):
                        result.append(json_string[i:i + 6])
                        i += 6
                        continue

                # Invalid escape: add an extra backslash
                result.append("\\\\")
                i += 1
                continue

            result.append(char)
            i += 1

        return "".join(result)

    def save_output(self, prompts_output: list[str]) -> None:
        """Validate the generated results and write them to the output file.

        If the JSON is invalid, the invalid escapes of the last result are
        fixed and the validation is tried again. The output directory is
        created if needed.

        Args:
            prompts_output: JSON objects (as text) generated so far.

        Raises:
            ValueError: If the JSON is still invalid after the fix.
        """

        def write_output(prompts_output: list[str]) -> str:
            """Join the generated objects into a JSON array text.

            Args:
                prompts_output: JSON objects (as text) to join.

            Returns:
                The JSON array as a string.
            """

            output = "["

            for prompt in prompts_output:
                output += prompt
                if prompt != prompts_output[-1]:
                    output += ","

            output += "\n]"

            return output

        output = write_output(prompts_output)

        if os.path.dirname(self.output_path) != "":
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        try:
            json.loads(output)
        except json.JSONDecodeError:
            prompts_output[-1] = self.fix_invalid_json_escapes(
                prompts_output[-1])
            output = write_output(prompts_output)
            try:
                json.loads(output)
                with open(self.output_path, "w") as file:
                    file.write(output)
                return
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {self.output_path}: {e}"
                ) from e

        with open(self.output_path, "w") as file:
            file.write(output)

    @staticmethod
    def print_time(initial_time: float, final_time: float, text: str) -> str:
        """Format an elapsed time as a readable message.

        Args:
            initial_time: Start timestamp, in seconds.
            final_time: End timestamp, in seconds.
            text: Label placed before the elapsed time.

        Returns:
            A message such as "Label: 1 minute and 3.20 seconds", preceded by
            a line break.
        """

        execution_time = final_time - initial_time
        exec_time_minutes = execution_time // 60
        exec_time_seconds = execution_time % 60

        if exec_time_minutes == 0:
            text = (f"\n{text}: {exec_time_seconds:.2f} "
                    f"{'second' if exec_time_seconds == 1 else 'seconds'}")
        else:
            text = (f"\n{text}: {exec_time_minutes:.0f} "
                    f"{'minute' if exec_time_minutes == 1 else 'minutes'} and "
                    f"{exec_time_seconds:.2f} "
                    f"{'second' if exec_time_seconds == 1 else 'seconds'}")

        return text

    def constrained_decoding(self) -> None:
        """Process every prompt and save the generated function calls.

        For each prompt the LLM first chooses the function name and then
        the parameters, both under constrained decoding. The output file
        is updated after every prompt and the progress is shown on the
        terminal.

        Raises:
            ValueError: If the generated JSON is invalid.
        """

        prompts_output: list[str] = []
        function_ids = self.valid_tokens_functions()
        self.terminal_output += "\nCall me maybe - output generation:\n"
        for index, prompt in enumerate(self.tests_json_data):
            initial_time = time()
            functions_json_data_prompt = deepcopy(self.functions_json_data)
            partial_prompt = get_prompt(
                functions_json_data_prompt, prompt.prompt)
            prompt_output = ("\n  {\n    \"prompt\": "
                             f"{json.dumps(prompt.prompt, ensure_ascii=False)}"
                             ",\n    \"name\": \"")
            if index == 0:
                self.terminal_output += "\n" + "=" * 130

            self.terminal_output += f"\n{index + 1}º prompt: {prompt.prompt}"
            print("\n" * 50 + self.terminal_output + prompt_output)

            # write the function name (llm choose the function):
            partial_prompt, prompt_output = self.write_function(
                partial_prompt,
                prompt_output,
                function_ids,
                functions_json_data_prompt,
                prompt.prompt)
            prompt_output += '",\n    "parameters": {'
            print("\n" * 50 + self.terminal_output + prompt_output)
            # write the parameters(llm choose the parameters):
            prompt_output = self.write_parameters(partial_prompt,
                                                  prompt_output,
                                                  functions_json_data_prompt)
            prompt_output += '\n  }'
            print("\n" * 50 + self.terminal_output + prompt_output)
            final_time = time()
            self.terminal_output += prompt_output
            self.terminal_output += self.print_time(
                initial_time, final_time, "Execution time")
            self.terminal_output += "\n" + "=" * 130
            print("\n" * 50 + self.terminal_output)

            prompts_output.append(prompt_output)
            self.save_output(prompts_output)

        print(f"\nFinal output saved in '{self.output_path}'")
