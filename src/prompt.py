
from typing import Any
from .get_data import ParameterType


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


# now prompt will receive each prompt and the functions_json_data att with the available functions
def get_prompt(functions_json_data: Any, prompt: str) -> str:

    prompt_text = (f"Write a function call json output as a \
dictionaire, follow exactly the example:\n\
\n{{\n  'prompt': prompt_text,\n  'name': function_name,\n\
  'parameters': {{first_parameter: parameter, \
second_parameter: parameter, ...}}\n}} \n\n\
Available functions: \n\n\
{functions_def(functions_json_data)} \n\n\
The function must follow this prompt:\n\n\
{prompt}")

    return prompt_text
