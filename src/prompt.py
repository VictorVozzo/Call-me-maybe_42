
from typing import Any
from .get_data import FunctionsDefinition, FunctionCalling, ParameterType


def function_calling(tests_json_data: Any) -> str:
    text = ""

    for test in tests_json_data:
        text += f"Prompt: {test.prompt}\n"

    return text


def functions_def_parameters(functions_parameter: dict[str, ParameterType]) -> str:
    text = ""

    for key, value in functions_parameter.items():
        text += f"{key}: {value.type}, "

    return text.rstrip(", ")


def functions_def(functions_json_data: Any) -> str:
    text = ""

    for function in functions_json_data:
        text += f"Function name: {function.name}, \
description: {function.description} , \
parameters: ({functions_def_parameters(function.parameters)}), \
return type: {function.returns.type}. \n"

    return text


def prompt(functions_json_data: Any, tests_json_data: Any) -> str:
    prompt = f"Write a function call json output as a \
list of dictionaires, follow exactly the example:\n\
[\n{{\n'prompt': prompt_text,\n'name': function_name,\n\
'parameters': {{first_parameter: parameter, \
second_parameter: parameter, ...}}\n }},\n ...\n] \n\n\
Available functions: \n\n\
{functions_def(functions_json_data)} \n\n\
The functions must follow this prompts:\n\n\
{function_calling(tests_json_data)}"

    return prompt
