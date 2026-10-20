# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    building.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vivozzo- <vivozzo-@student.42porto.com>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/10/20 12:20:56 by vivozzo-          #+#    #+#              #
#    Updated: 2026/10/20 15:21:31 by vivozzo-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from typing import Any


def prompt(functions_json_data: Any, tests_json_data: Any) -> str:
    prompt = f"Write a function call json output as a \
list of dictionaires, follow exactly the example:\n \
[{{'prompt': prompt_text, 'name': function_name, \
'parameters': {{first_parameter: parameter, \
second_parameter: parameter, ...}} }}, ...] \n\nYou must use this functions: \n\n\
{functions_json_data} \n\nThe functions must follow this prompts: \n\n \
{tests_json_data}"

    return prompt
