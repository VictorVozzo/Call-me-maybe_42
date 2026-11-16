*This project has been created as part of the 42 curriculum by vivozzo-.*

# Call Me Maybe — Function Calling with Constrained Decoding

## Description

**Call Me Maybe** is a function calling tool that translates natural language
prompts into structured, machine-executable function calls, using a small
language model (**Qwen/Qwen3-0.6B**, ~0.6B parameters).

Given a request such as `"What is the sum of 40 and 2?"`, the program does
**not** answer `42`. Instead it produces the call that should be executed:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

Small models are unreliable at producing valid JSON when they are only asked
to do so in a prompt. This project solves that with **constrained decoding**:
the model never gets to write free text. At every generation step the set of
tokens it may choose from is restricted to the ones that keep the output valid
and compliant with the schema of `functions_definition.json`. The function to
call and the value of every argument are still chosen by the LLM, but the
structure (braces, keys, quotes, commas, types) is guaranteed by the program.

### Overview

1. Read and validate the function definitions and the prompts (pydantic).
2. Load the model vocabulary and pre-compute which tokens are valid inside a
   number, a string or a boolean.
3. For each prompt, let the LLM choose the **function name** among the real
   function names only.
4. Let the LLM generate each **parameter**, respecting the type declared for
   it (`number`/`integer`/`float`, `boolean`, `string`).
5. Validate the JSON and write `function_calls.json` after every prompt.

## Instructions

### Requirements

- Python **3.10+**
- [`uv`](https://docs.astral.sh/uv/) (the `Makefile` installs it if missing)
- Internet access on the **first** run, so the Qwen3-0.6B weights can be
  downloaded from the Hugging Face Hub (they are cached afterwards in
  `.cache/huggingface`, see below)

### Installation

```bash
make install        # or simply: uv sync
```

This creates the virtual environment and installs `numpy`, `pydantic`,
`accelerate` and the local `llm_sdk` package.

### Execution

```bash
make run
```

or, directly:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

The program accepts either **no arguments** (the default paths above are used)
or **all three** options, in the order shown. Anything else prints a usage
message.

> The `Makefile` exports `HF_HOME=$(PROJECT_DIR)/.cache/huggingface`, so the
> model is cached inside the project. When running `uv run ...` by hand, the
> default Hugging Face cache of your machine is used instead.

### Makefile rules

| Rule          | What it does                                                          |
|---------------|-----------------------------------------------------------------------|
| `install`     | Installs `uv` if needed and runs `uv sync`                            |
| `run`         | Runs the program with the default input/output paths                  |
| `debug`       | Runs the program under `pdb`                                          |
| `lint`        | `flake8` + `mypy` with the flags required by the subject              |
| `lint-strict` | `flake8` + `mypy --strict`                                            |
| `clean`       | Removes `__pycache__`, `.mypy_cache` and `*.pyc`                      |
| `fclean`      | `clean` + removes `.venv` and `.cache`                                |
| `re`          | `fclean` followed by `install`                                        |

### Project structure

```
.
├── Makefile
├── README.md
├── pyproject.toml / uv.lock
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
└── src/
    ├── __main__.py              # entry point, global error handling
    ├── get_data.py              # CLI parsing + pydantic models + JSON loading
    ├── prompt.py                # builds the prompt sent to the LLM
    ├── constrained_decoding.py  # the constrained decoding engine
    └── llm_sdk/                 # provided SDK (Small_LLM_Model)
```

`data/output/` is generated at run time and is not part of the repository.

## Example usage

Default paths:

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
    --functions_definition my_functions.json \
    --input my_prompts.json \
    --output results/calls.json
```

Available functions (`functions_definition.json`). Each function has
a `name`, a `description`, its `parameters` (name and type) and its `returns`
type:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {"type": "string"}
    },
    "returns": {"type": "string"}
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": {"type": "string"},
      "regex": {"type": "string"},
      "replacement": {"type": "string"}
    },
    "returns": {"type": "string"}
  }
]
```

Input prompts (`function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Expected shape of the output (`function_calls.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

While running, the terminal shows a live view of the generation, token by
token, followed by the execution time of each prompt and the total time.

Error handling examples (the program prints a clear message and does not
crash):

- missing file → `File not found: <path>`
- malformed JSON → `Invalid JSON in <path>: ...`
- file that is not a non-empty array, or items that do not match the expected
  schema → a validation error message
- wrong command line usage → a usage message

## Algorithm explanation

The implementation lives in `src/constrained_decoding.py`
(class `ConstrainedDecoding`).

### 1. Background: logit masking

At every step the model returns one logit per vocabulary token. Constrained
decoding means that, before choosing the next token, every token that would
break the structure or the schema is excluded (conceptually, its logit is set
to `-inf`). Since this project uses **greedy decoding**, this is implemented
as an `argmax` over the logits of the *allowed* token ids only
(`find_max_logit_id`), which gives exactly the same result as masking with
`-inf` and then taking the maximum.

### 2. Pre-computation (done once, at start-up)

- **Vocabulary**: the file returned by `get_path_to_vocab_file()` maps
  `token → id`. It is inverted into `id → token` so any token id can be
  checked as a string.
- **Number tokens** (`get_valid_tokens_numbers`): every token that can be part
  of a JSON number is classified. Four lists are produced: tokens valid for a
  parameter followed by `,`, tokens valid for the *last* parameter (followed by
  `}`), negative-number tokens (including a lone `-`) and tokens that contain a
  decimal point. Tokens that fuse a number with its terminator (e.g. `5,` or
  `5}`) are accepted too, because they close the parameter.
- **String tokens** (`get_valid_tokens_string`): tokens without a double quote
  are always valid inside a string. A token with a single closing quote is only
  valid if nothing follows it or if it is followed by `,` (middle parameters)
  or `}` (last parameter).
- **Boolean tokens**: `true` and `false` are encoded with the tokenizer.
- **Function names**: every function name is encoded into its list of token
  ids.

### 3. Generation of one function call

The output object is assembled piece by piece. The **fixed structure** is
written by the program; only the *choices* are made by the LLM:

```
{ "prompt": <copied from the input, JSON-escaped>,
  "name": "<LLM choice, restricted to real function names>",
  "parameters": {"<key from schema>": <LLM value>, ...} }
```

**a) Function name** (`write_function`). All function names are treated like
a small trie of token ids. At step *i* the allowed tokens are the *i*-th
tokens of the candidate names that are still compatible with what was already
generated. The LLM picks the allowed token with the highest logit, the
candidates that do not match are discarded, and the prompt is rebuilt so it
only lists the remaining candidates. When a single candidate remains, its
remaining tokens are appended directly, without calling the LLM again.

**b) Parameters** (`write_parameters`). The keys, quotes, colons, commas and
closing braces come from the schema, not from the model. Only the values are
generated, using the token set that matches the declared type:

- **number / integer / float**: only numeric tokens are allowed (plus `-` on
  the first parameter). Once a decimal point has been generated (or if the type
  is `integer`), tokens containing a point are removed from the allowed set, so
  a number can never contain two points. A terminator token (`,` or `}`) ends
  the value. Non-integer numbers without a decimal part are rewritten as
  `X.0`, so `2` becomes `2.0`, as required by the subject.
- **boolean**: the LLM chooses between the first token of `true` and the first
  token of `false`; the program then writes the rest of the literal.
- **string**: the opening quote is written by the program; then any token
  without a double quote is allowed, and a token that closes the string (with
  the right terminator for a middle or last parameter) ends the value. Typographic
  quotes (`“ ” « » ...`) generated as a closing quote are normalized to `"`.
  Hard limits (32 tokens for numbers, 64 for strings) prevent endless loops.

**c) Safety net** (`save_output`, `fix_invalid_json_escapes`). The result is
checked with `json.loads` before it is written. Strings such as regular
expressions (`\d+`) may contain backslashes that are not valid JSON escapes;
if the validation fails, every invalid backslash of the last result is escaped
and the validation is tried again. If the JSON is still invalid, a clear error
is raised.

### 4. The prompt

The text sent to the LLM (`prompt.py`) contains the list of available
functions (name, description, parameters, return type), an example of the
expected JSON and the user request, and tells the model to choose the function
that matches the request without solving it. The prompt only gives the model
*context*; the validity of the output never depends on the model following
it, because every token is constrained.

## Design decisions

- **Greedy decoding over allowed tokens.** Deterministic output (the same input
  always gives the same result) and simple to reason about and test.
- **The program writes the structure, the LLM writes the content.** Keys,
  punctuation and types never depend on the model, which is what makes the
  JSON validity independent of the model size.
- **The input prompt is copied, not generated.** The `prompt` field is
  inserted with `json.dumps`, so it always matches the input exactly and is
  correctly escaped.
- **The function is chosen by the LLM, with no heuristics.** The candidate
  names are the only allowed continuations, and the model chooses among them
  token by token. No keyword matching or hard-coded rules are used.
- **Candidate pruning.** As soon as a function name is no longer possible, it
  disappears from the allowed tokens *and* from the prompt, which helps a
  small model to focus. When only one candidate is left the LLM is skipped.
- **Pre-computed token sets.** Classifying the whole vocabulary is done once
  at start-up, not at every step.
- **Pydantic for all data models** (`FunctionCalling`, `ParameterType`,
  `FunctionsDefinition`) with `strict=True` and `extra="forbid"`, so malformed
  input files are rejected with a clear message.
- **The output file is rewritten after every prompt**, so partial results are
  kept if something goes wrong later.
- **Only the public API of `llm_sdk` is used** (`encode`, `decode`,
  `get_logits_from_input_ids`, `get_path_to_vocab_file`).
- **A global `try/except` in `main()`** turns any error into a readable
  message instead of a traceback.

## Performance analysis

- **Reliability.** The structure of the output is enforced by the program, and
  the final text is validated with `json.loads` before being written, so the
  output file is always parseable JSON with the exact keys `prompt`, `name`
  and `parameters`, and typed values matching the schema.
- **Accuracy.** Choosing the function and extracting the arguments are left to
  the model, so accuracy depends on Qwen3-0.6B; constrained decoding removes
  the formatting errors but cannot make the model understand a prompt it does
  not understand. Pruning the candidate functions from the prompt helps the
  choice.
- **Speed.** The cost is dominated by LLM forward passes: one per generated
  value token (plus one per function-name token while more than one candidate
  remains). The SDK recomputes the whole sequence at each call (no key/value
  cache), so the time grows with the prompt length and the length of the
  values. Skipping the LLM when only one function remains, and pre-computing
  the token sets, reduce the work.

### Measured results

Measured on the 11 prompts of `data/input/function_calling_tests.json`:

| Machine | Prompts | Total time | Average per prompt | Correct calls |
|---------|---------|------------|--------------------|---------------|
| Intel Core i7-10700 @ 2.90 GHz (8 cores / 16 threads), 15 GiB RAM | 11 | 1 min 40 s – 1 min 50 s | ~9–10 s | 10 / 11 (≈ 91 %) |

"Correct call" means that both the function and all its arguments were the
expected ones. These results meet the targets of the subject (90%+ accuracy
and all prompts processed in under 5 minutes). The total time varies by a few
seconds between runs, depending on the load of the machine. Output validity
was 100%: every run produced a parseable file that followed the schema.

## Challenges faced

- **Numbers fused with delimiters.** BPE vocabularies contain tokens like
  `5,` or `.5}` in addition to single digits. Every token had to be classified
  by checking an optional leading `-`, an optional trailing `,`/`}` and an
  optional decimal point, so that a parameter can be closed correctly without
  generating an invalid number.
- **Integers vs floats.** The subject requires numbers as floats (`2.0`). The
  decoder appends `.0` when the model stops after an integer value, and removes
  the decimal-point tokens for `integer` parameters and after the first point.
- **Closing a string at the right moment.** A string ends with a quote, but
  that quote can be a separate token or merged with `,` or `}`. Different token
  sets are used for middle and last parameters. The model also sometimes tries
  to close a string with a typographic quote, which is normalized to `"`.
- **Invalid escapes in regular expressions.** Arguments for functions like
  `fn_substitute_string_with_regex` contain backslashes (`\d`, `\s`), which are
  not valid JSON escapes. This is fixed by a post-validation step that escapes
  them.
- **Function names split into several tokens.** Names are tokenized into
  several ids, so the decoder tracks the candidates token by token instead of
  comparing whole strings.
- **Readability of the generation.** A live terminal view was added to follow
  the generation token by token while debugging.

## Testing strategy

The project is tested end to end by running it and checking the produced
file against the schema of `functions_definition.json`:

1. Run the program on the provided `data/input/` files and check that
   `data/output/function_calls.json` is created.
2. Load the output with a JSON parser (no trailing commas, no extra keys,
   no prose) and verify, for every entry, that the keys are exactly `prompt`,
   `name` and `parameters`, that `name` is one of the defined functions and
   that every parameter exists and has the declared type.
3. Manually check that the chosen function and the extracted arguments are the
   expected ones for each prompt.
4. Edge cases, as suggested by the subject: empty strings, large and negative
   numbers, special characters and quotes, regular expressions with
   backslashes, ambiguous prompts, and functions with several parameters.
5. Error handling: missing input files, invalid JSON, an empty array, items
   with unknown or missing keys, and wrong command line arguments must all give
   a clear message and never a traceback.
6. Static checks: `make lint` (`flake8` and `mypy` with the required flags) and
   optionally `make lint-strict`.

## Resources

**Documentation and references**

- [Qwen3 model card (`Qwen/Qwen3-0.6B`)](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face — Tokenizers and byte-level BPE](https://huggingface.co/docs/tokenizers)
- [RFC 8259 — The JSON Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/),
  [flake8](https://flake8.pycqa.org/), [mypy](https://mypy.readthedocs.io/)

**How AI was used**

AI tools were used as a learning and review aid, not as a code generator.

- **Learning concepts**: to understand the ideas behind the project, such as
  how an LLM turns tokens into logits, how the tokenizer and the vocabulary
  file map text to token ids, how constrained decoding masks invalid tokens,
  and why this guarantees valid, schema-compliant JSON even with a small model.
- **Code review**: to review the code I had already written, point out
  possible bugs and edge cases, and suggest improvements
  to readability, docstrings and type hints, which I then checked, tested and
  applied myself.
- **README**: to help structure and organize this README. The technical
  content was checked against the actual code and the real program output.

No part of the final solution was copied from an AI tool without being
understood first.
