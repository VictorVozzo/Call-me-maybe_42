# *************************************************************************** #
#                                                                             #
#                                                         :::      ::::::::   #
#   get_data.py                                         :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#   By: vivozzo- <vivozzo-@student.42porto.com>     +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#   Created: 2026/11/04 16:38:08 by vivozzo-           #+#    #+#             #
#   Updated: 2026/11/06 10:39:24 by vivozzo-          ###   ########.fr       #
#                                                                             #
# *************************************************************************** #

import json
from typing import Any, TypeVar
from pydantic import BaseModel, ConfigDict, ValidationError


TV = TypeVar("TV", bound=BaseModel)


class FunctionCalling(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: str


class ParameterType(BaseModel):
    model_config = ConfigDict(strict=True)

    type: str


class FunctionsDefinition(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType


def load_json(file_path: str, model_class: type[TV]) -> list[TV]:

    try:
        with open(file_path, "r") as file:
            content = json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"File not found: {file_path}"
        ) from e
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {file_path}: {e}"
        ) from e

    if not isinstance(content, list):
        raise ValueError(
            f"{file_path} must contain a JSON array"
        )

    if not content:
        raise ValueError(
            f"{file_path} is empty"
        )

    errors = ""
    list_model_class = []

    for item in content:
        try:
            item_validate = model_class.model_validate(item)
            list_model_class.append(item_validate)
        except ValidationError as e:
            errors += f"Validation error in {file_path}: {e}\n\n"

    if errors != "":
        raise Exception(errors)

    return list_model_class


def parsing(argv: list[str]) -> tuple[list[FunctionsDefinition], list[FunctionCalling], str]:

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

    functions_json_data = load_json(
        functions_definition_path, FunctionsDefinition)
    tests_json_data = load_json(input_path, FunctionCalling)

    return (functions_json_data, tests_json_data, output_path)
