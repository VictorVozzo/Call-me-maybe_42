# *************************************************************************** #
#                                                                             #
#                                                         :::      ::::::::   #
#   prompt.py                                           :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#   By: vivozzo- <vivozzo-@student.42porto.com>     +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#   Created: 2026/11/16 17:08:29 by vivozzo-           #+#    #+#             #
#   Updated: 2026/11/16 17:25:31 by vivozzo-          ###   ########.fr       #
#                                                                             #
# *************************************************************************** #


from typing import Any
from .get_data import ParameterType


def functions_def_parameters(functions_parameter: dict[
        str, ParameterType]) -> str:
    """Format the parameters of a function as a readable string.

    Args:
        functions_parameter: Mapping of parameter name to its ParameterType.

    Returns:
        A string such as ``"a: number, b: number"`` (no trailing comma).
    """

    text = ""

    for key, value in functions_parameter.items():
        text += f"{key}: {value.type}, "

    return text.rstrip(", ")


def functions_def(functions_json_data: Any) -> str:
    """Describe every available function in plain text for the prompt.

    Args:
        functions_json_data: Iterable of FunctionsDefinition objects.

    Returns:
        One line per function with its name, description, parameters and
        return type.
    """

    text = ""

    for function in functions_json_data:
        text += f"Function name: {function.name}, \
description: {function.description} , \
parameters: ({functions_def_parameters(function.parameters)}), \
return type: {function.returns.type}. \n"

    return text


def get_prompt(functions_json_data: Any, prompt: str) -> str:
    """Build the full prompt that is sent to the LLM for a single request.

    The prompt instructs the model to pick the best matching function
    (without solving the request), lists the available functions, shows
    an example of the expected JSON and appends the user request.

    Args:
        functions_json_data: Iterable of FunctionsDefinition objects that
            are still available for the current request.
        prompt: The natural language request written by the user.

    Returns:
        The complete prompt text.
    """

    prompt_text = (f"You are a precise function calling assistant.\
Pick the function that best matches the user request.\
Do not solve the request.\
Respond or complete ONLY with a VALID raw JSON string.\
All numbers must be float.\nAvailable functions: \n\n\
{functions_def(functions_json_data)} \n\n\
Example:\n\
\n  {{\n    \"prompt\": \"prompt_text\",\n    \"name\":   \"function_name\",\
\n    \"parameters\": {{\"first_parameter\": \"first_parameter_value\", \
\"second_parameter\": \"second_parameter_value\"}}\n  }} \n\n\
User request: \
\"{prompt}\"")

    return prompt_text
