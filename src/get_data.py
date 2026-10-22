
import json
from typing import Any


def read_input(functions_definition_path: str,
               function_calling_tests: str) -> tuple[str, tuple[Any]]:

    errors = ""

    try:
        with open(functions_definition_path, "r") as file:
            content = file.read()

            if not content.strip():
                errors += "functions_definition.json file is empty\n"
            else:
                functions_json_data = json.loads(content)
    except FileNotFoundError as e:
        errors += f"Error: {e}\n"
    except json.decoder.JSONDecodeError as error:
        errors += f"json error in functions_definition file: {error}\n"

    try:
        with open(function_calling_tests, "r") as f:
            content = f.read()

            if not content.strip():
                errors += "function_calling_tests.json file is empty\n"
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
    error_msg = ("function_calling_tests.json invalid json syntax, "
                 "must be:\n"
                 "[{'prompt': prompt_text}, ...]\n"
                 "prompt_text need to be a string")

    if not (isinstance(tests_json_data, list) and
            all(isinstance(item, dict) for item in tests_json_data)):
        return error_msg

    for test in tests_json_data:
        if len(test) != 1:
            return error_msg
        if 'prompt' not in test:
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
        if ('name' not in function or 'description' not in function or
                'parameters' not in function or 'returns' not in function):
            return error_msg
        if (not isinstance(function['name'], str) or
                not isinstance(function['description'], str)):
            return error_msg
        if not isinstance(function['parameters'], dict):
            return error_msg
        for parameter_value in function['parameters'].values():
            if not isinstance(parameter_value, dict):
                return error_msg
            if len(parameter_value) != 1:
                return error_msg
            if ('type' not in parameter_value or
                    not isinstance(parameter_value['type'], str)):
                return error_msg
        if not isinstance(function['returns'], dict):
            return error_msg
        if len(function['returns']) != 1:
            return error_msg
        if ('type' not in function['returns'] or
                not isinstance(function['returns']['type'], str)):
            return error_msg

    return ""


def parsing(argv: list[str]) -> tuple[Any]:
    errors = ""

    if len(argv) != 1 and len(argv) != 7:
        raise Exception("Usage: v run python -m src "
                        "[--functions_definition <function_definition_file>] "
                        "[--input <input_file>] [--output <output_file>] "
                        "arguments in square brackets are optional")

    if (len(argv) == 7 and (argv[1] != "--functions_definition" or
                            argv[3] != "--input" or
                            argv[5] != "--output")):
        raise Exception("Usage: v run python -m src "
                        "[--functions_definition <function_definition_file>] "
                        "[--input <input_file>] [--output <output_file>] "
                        "arguments in square brackets are optional")

    if len(argv) == 1:
        functions_definition_path = "/data/input/functions_definition.json"
        input_path = "/data/input/function_calling_tests.json"
        output_path = "/data/output/function_calls.json"
    else:
        functions_definition_path = argv[2]
        input_path = argv[4]
        output_path = argv[6]

    read = read_input(functions_definition_path, input_path)
    errors = read[0]
    if errors:
        raise Exception(errors)

    functions_json_data, tests_json_data = read[1]
    errors += parsing_function_calling_tests(tests_json_data)
    errors += parsing_functions_definition(functions_json_data)

    if errors:
        raise Exception(errors)
    return (functions_json_data, tests_json_data)
