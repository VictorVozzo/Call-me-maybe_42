# *************************************************************************** #
#                                                                             #
#                                                         :::      ::::::::   #
#   __main__.py                                         :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#   By: vivozzo- <vivozzo-@student.42porto.com>     +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#   Created: 2026/11/16 17:08:07 by vivozzo-           #+#    #+#             #
#   Updated: 2026/11/16 18:37:02 by vivozzo-          ###   ########.fr       #
#                                                                             #
# *************************************************************************** #


from sys import argv
from .get_data import parsing
from . constrained_decoding import ConstrainedDecoding
from time import time


def main() -> None:
    """Run the whole pipeline: parse arguments, decode and save the output.

    Reads the command line arguments, loads and validates the input files,
    runs the constrained decoding over every prompt and prints the total
    execution time. Any exception is caught and reported as a clear message
    instead of crashing the program.

    Returns:
        None
    """
    try:
        initial_time = time()

        parsing_data = parsing(argv)
        functions_json_data, tests_json_data, output_path = parsing_data

        constrained_decoding = ConstrainedDecoding(
            functions_json_data, tests_json_data, output_path)
        constrained_decoding.constrained_decoding()
        final_time = time()
        print(ConstrainedDecoding.print_time(
            initial_time, final_time, "\n\nFinal execution time"))
    except Exception as e:
        print(f"Caught an error: \n{e}")


if __name__ == "__main__":
    main()
