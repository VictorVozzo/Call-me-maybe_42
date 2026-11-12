
from json import dumps
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

    prompt_text = (f"You are a precise function calling assistant. Pick the function that best matches \
the user request. Do not solve the request. Respond or complete ONLY with a VALID raw JSON string.NO identations of any type.\nAvailable functions: \n\n\
{functions_def(functions_json_data)} \n\n\
Example:\n\
\n  {{\n    \"prompt\":   \"prompt_text\",\n    \"name\":   \"function_name\",\n\
    \"parameters\":   {{\"first_parameter\": \"first_parameter_value\", \
\"second_parameter\": \"second_parameter_value\"}}\n  }} \n\n\
User request: \
\"{prompt}\"")

    return prompt_text
