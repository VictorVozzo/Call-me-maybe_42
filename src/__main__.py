

from sys import argv
from .get_data import parsing
from .building import prompt


def main() -> None:
    try:
        parsing_data = parsing(argv)
        functions_json_data, tests_json_data = parsing_data
        # print(functions_json_data)
        # print("\n\n")
        # print(tests_json_data)
        print(prompt(functions_json_data, tests_json_data))
    except Exception as e:
        print(f"Caught an error: \n{e}")


if __name__ == "__main__":
    main()
