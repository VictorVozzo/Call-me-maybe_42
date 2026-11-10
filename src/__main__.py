# *************************************************************************** #
#                                                                             #
#                                                         :::      ::::::::   #
#   __main__.py                                         :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#   By: vivozzo- <vivozzo-@student.42porto.com>     +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#   Created: 2026/11/04 17:09:30 by vivozzo-           #+#    #+#             #
#   Updated: 2026/11/10 12:02:41 by vivozzo-          ###   ########.fr       #
#                                                                             #
# *************************************************************************** #

from sys import argv
from .get_data import parsing
from .prompt import prompt
from . constrained_decoding import ConstrainedDecoding
from .llm_sdk.llm_sdk import Small_LLM_Model
from numpy import argmax


def main() -> None:
    try:
        parsing_data = parsing(argv)
        functions_json_data, tests_json_data, output_path = parsing_data
        prompt_list = prompt(functions_json_data, tests_json_data)
        constrained_decoding = ConstrainedDecoding(functions_json_data,tests_json_data, prompt_list)
        constrained_decoding.constrained_decoding()
        # print(constrained_decoding.id_content_vocab)
    except Exception as e:
        print(f"Caught an error: \n{e}")


if __name__ == "__main__":
    main()
