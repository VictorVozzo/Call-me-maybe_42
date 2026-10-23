

from sys import argv
from .get_data import parsing
from .prompt import prompt
from .llm_sdk.llm_sdk import Small_LLM_Model
from numpy import argmax

def main() -> None:
    try:
        parsing_data = parsing(argv)
        functions_json_data, tests_json_data = parsing_data
        # print(functions_json_data)
        # print("\n\n")
        # print(tests_json_data)
        prompt_text = prompt(functions_json_data, tests_json_data)
        small_llm = Small_LLM_Model()
        ids = small_llm.encode("What is the result of 1 + 4 ?").tolist()[0]
        logits = small_llm.get_logits_from_input_ids(ids)
        best_token = argmax(logits)
        next_token = small_llm.decode(best_token)
        print(small_llm.get_path_to_vocab_file())

        

        

    except Exception as e:
        print(f"Caught an error: \n{e}")


if __name__ == "__main__":
    main()
