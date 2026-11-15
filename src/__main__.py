# *************************************************************************** #
#                                                                             #
#                                                         :::      ::::::::   #
#   __main__.py                                         :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#   By: vivozzo- <vivozzo-@student.42porto.com>     +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#   Created: 2026/11/04 17:09:30 by vivozzo-           #+#    #+#             #
#   Updated: 2026/11/12 17:00:06 by vivozzo-          ###   ########.fr       #
#                                                                             #
# *************************************************************************** #

from sys import argv
from .get_data import parsing
from . constrained_decoding import ConstrainedDecoding
from time import time


def main() -> None:
    try:
        initial_time = time()

        parsing_data = parsing(argv)
        functions_json_data, tests_json_data, output_path = parsing_data
        
        constrained_decoding = ConstrainedDecoding(
            functions_json_data, tests_json_data, output_path)
        constrained_decoding.constrained_decoding()
        final_time = time()
        print(ConstrainedDecoding.print_time(initial_time, final_time, f"\n\nFinal execution time"))
    except Exception as e:
        print(f"Caught an error: \n{e}")


if __name__ == "__main__":
    main()
