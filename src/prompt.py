
from typing import Any


def function_calling(tests_json_data: Any) -> str:
    text = ""

    for test in tests_json_data:
        text += f"Prompt: {test['prompt']}\n"

    return text


def functions_def(functions_json_data: Any) -> str:
    text = ""

    for function in functions_json_data:
        text += f"Function name: {function['name']}, \
description: {function['description']}, \
parameters: {function['parameters']}\n"

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
