
import json
from typing import Any


def read_input() -> tuple[str, tuple[Any]]:

    errors = ""

    try:
        with open("../data/input/functions_definition.json", "r") as file:
            content = file.read()

            if not content.strip():
                errors += f"functions_definition.json file is empty\n"
            else:
                functions_json_data = json.loads(content)
    except FileNotFoundError as e:
        errors += f"Error: {e}\n"
    except json.decoder.JSONDecodeError as error:
        errors += f"json error in functions_definition file: {error}\n"

    try:
        with open("../data/input/function_calling_tests.json", "r") as f:
            content = f.read()

            if not content.strip():
                errors += f"function_calling_tests.json file is empty\n"
            else:
                tests_json_data = json.loads(content)
    except FileNotFoundError as e:
        errors += f"Error: {e}\n"
    except json.decoder.JSONDecodeError as error:
        errors += f"json error in function_calling_tests file: {error}\n"

    if errors:
        return (errors, ())

    return (errors, (functions_json_data, tests_json_data))


def parsing_function_calling_tests(tests_json_data: Any) -> str:
    error_msg = ("tests_json_data.json invalid json syntax, "
                 "must be:\n"
                 "[{'prompt': prompt_text}, ...]\n"
                 "prompt_text need to be a string")

    if not (isinstance(tests_json_data, list) and
            all(isinstance(item, dict) for item in tests_json_data)):
        return error_msg

    for test in tests_json_data:
        if len(test) != 1:
            return error_msg
        if not 'prompt' in test:
            return error_msg
        if not isinstance(test['prompt'], str):
            return error_msg

    return ""


def parsing_functions_definition(functions_json_data: Any) -> str:
    error_msg = ("functions_definition.json invalid json syntax, "
                 "must be:\n"
                 "[{'name': function name}, \n"
                 "{'description: function description}, \n"
                 "{'parameters': {parameter: {'type': parameter type}, ...} }"
                 ", \n{'returns': {'type': variable type} }, ...]\n"
                 "function name, function description, parameter and"
                 " function/parameter type need to be a string\n")

    if not (isinstance(functions_json_data, list) and
            all(isinstance(item, dict) for item in functions_json_data)):
        return error_msg

    for function in functions_json_data:
        if len(function) != 4:
            return error_msg
        if (not 'name' in function or not 'description' in function or
                not 'parameters' in function or not 'returns' in function):
            return error_msg
        if not isinstance(function['name'], str) or not isinstance(function['description'], str):
            return error_msg
        if not isinstance(function['parameters'], dict):
            return error_msg
        for parameter_value in function['parameters'].values():
            if not isinstance(parameter_value, dict):
                print("5: entrou aqui")
                return error_msg
            if len(parameter_value) != 1:
                print("6: entrou aqui")
                return error_msg
            if not 'type' in parameter_value or not isinstance(parameter_value['type'], str):
                print("7: entrou aqui")
                return error_msg
        if not isinstance(function['returns'], dict):
            return error_msg
        if len(function['returns']) != 1:
            return error_msg
        if not 'type' in function['returns'] or not isinstance(function['returns']['type'], str):
            return error_msg

    return ""

def parsing() -> tuple[str, tuple[Any]]:
    errors = ""
    read = read_input()
    errors = read[0]
    if errors:
        return (errors, (None, None))

    functions_json_data, tests_json_data = read[1]
    errors += parsing_function_calling_tests(tests_json_data)
    errors += parsing_functions_definition(functions_json_data)

    return (errors,(functions_json_data, tests_json_data))


