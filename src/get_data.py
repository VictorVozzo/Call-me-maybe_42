# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    get_data.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vivozzo- <vivozzo-@student.42porto.com>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/10/20 12:20:40 by vivozzo-          #+#    #+#              #
#    Updated: 2026/10/20 13:41:01 by vivozzo-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import json
from typing import Any


def read_input(functions_definition_path: str,
               function_calling_tests: str) -> tuple[str, tuple[Any]]:

    errors = ""

    try:
        with open(functions_definition_path, "r") as file:
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
        with open(function_calling_tests, "r") as f:
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


def parsing() -> tuple[Any]:
    errors = ""
    read = read_input("../data/input/functions_definition.json",
                      "../data/input/function_calling_tests.json")
    errors = read[0]
    if errors:
        raise Exception (errors)

    functions_json_data, tests_json_data = read[1]
    errors += parsing_function_calling_tests(tests_json_data)
    errors += parsing_functions_definition(functions_json_data)

    if errors:
        raise Exception (errors)
    return (functions_json_data, tests_json_data)
