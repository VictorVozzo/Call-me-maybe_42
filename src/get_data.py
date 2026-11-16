# *************************************************************************** #
#                                                                             #
#                                                         :::      ::::::::   #
#   get_data.py                                         :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#   By: vivozzo- <vivozzo-@student.42porto.com>     +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#   Created: 2026/11/16 17:08:24 by vivozzo-           #+#    #+#             #
#   Updated: 2026/11/16 17:34:29 by vivozzo-          ###   ########.fr       #
#                                                                             #
# *************************************************************************** #


import json
from typing import TypeVar
from pydantic import BaseModel, ConfigDict, ValidationError


TV = TypeVar("TV", bound=BaseModel)


class FunctionCalling(BaseModel):
    """A natural language request to be converted into a function call.

    Attributes:
        prompt: The user request text.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str


class ParameterType(BaseModel):
    """Type description of a function parameter.

    Attributes:
        type: Type name as written in the definitions file
            (e.g. ``"number"``, ``"string"``, ``"boolean"``).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    type: str


class FunctionsDefinition(BaseModel):
    """Definition of a function that the LLM is allowed to call.

    Attributes:
        name: Function name.
        description: Short description of what the function does.
        parameters: Mapping of parameter name to its ParameterType.
        returns: Return type of the function.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType


def load_json(file_path: str, model_class: type[TV]) -> list[TV]:
    """Load a JSON array file and validate each item with a pydantic model.

    Args:
        file_path: Path of the JSON file to read.
        model_class: Pydantic model class used to validate every item.

    Returns:
        A list with one validated model instance per item of the array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the content is not valid JSON, is not an array or
            is an empty array.
        Exception: If one or more items fail validation (all validation
            errors are gathered in the message).
    """

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


def parsing(argv: list[str]) -> tuple[list[FunctionsDefinition],
                                      list[FunctionCalling], str]:
    """Parse the command line arguments and load the input files.

    The program accepts either no arguments (default paths are used) or
    all three options: ``--functions_definition``, ``--input`` and
    ``--output``, in this order.

    Args:
        argv: The raw argument list (``sys.argv``), including the program
            name at index 0.

    Returns:
        A tuple ``(functions, tests, output_path)`` with the validated
        function definitions, the validated prompts and the output file
        path.

    Raises:
        Exception: If the arguments do not follow the expected usage.
        FileNotFoundError: If an input file is missing.
        ValueError: If an input file is not a valid, non-empty JSON array.
    """

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
        functions_definition_path = "data/input/functions_definition.json"
        input_path = "data/input/function_calling_tests.json"
        output_path = "data/output/function_calls.json"
    else:
        functions_definition_path = argv[2]
        input_path = argv[4]
        output_path = argv[6]

    functions_json_data = load_json(
        functions_definition_path, FunctionsDefinition)
    tests_json_data = load_json(input_path, FunctionCalling)

    return (functions_json_data, tests_json_data, output_path)
