# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    main.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: vivozzo- <vivozzo-@student.42porto.com>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/10/20 12:20:45 by vivozzo-          #+#    #+#              #
#    Updated: 2026/10/20 13:40:59 by vivozzo-         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from get_data import parsing
from building import prompt

def main() -> None:
    try:
        parsing_data = parsing()
        functions_json_data, tests_json_data = parsing_data
        # print(functions_json_data)
        # print("\n\n")
        # print(tests_json_data)
        print(prompt(functions_json_data, tests_json_data))
    except Exception as e:
        print(f"Caught an error: {e}")

    

if __name__ == "__main__":
    main()
