import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-5.5"

# USER_REQUEST = """
# We expect to make 250 API requests.
# Each request will contain approximately 3,000 input tokens
# and produce approximately 400 output tokens.

# Calculate the total input tokens, total output tokens,
# and combined token volume.
# """

USER_REQUEST = """
We are planning 1,200 API calls per day.
Each contains roughly 8,500 input tokens and generates
around 750 output tokens.

How many tokens will we process in 30 days?
"""


def calculate_token_volume(
    number_of_requests: int,
    input_tokens_per_request: int,
    output_tokens_per_request: int,
) -> dict:
    """Calculate total token usage across multiple API requests."""

    total_input_tokens = number_of_requests * input_tokens_per_request
    total_output_tokens = number_of_requests * output_tokens_per_request

    return {
        "number_of_requests": number_of_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }


TOOLS = [
    {
        "type": "function",
        "name": "calculate_token_volume",
        "description": (
            "Calculate total input, output, and combined token volume "
            "across multiple API requests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "number_of_requests": {
                    "type": "integer",
                    "description": "Number of API requests.",
                },
                "input_tokens_per_request": {
                    "type": "integer",
                    "description": ("Average number of input tokens in each request."),
                },
                "output_tokens_per_request": {
                    "type": "integer",
                    "description": (
                        "Average number of output tokens generated per request."
                    ),
                },
            },
            "required": [
                "number_of_requests",
                "input_tokens_per_request",
                "output_tokens_per_request",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def load_environment() -> None:
    """Load the repository-level environment file."""

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


def execute_function_call(function_name: str, arguments: dict) -> dict:
    """Route a requested tool call to the corresponding Python function."""

    if function_name == "calculate_token_volume":
        return calculate_token_volume(**arguments)

    raise ValueError(f"Unknown function requested: {function_name}")


def main() -> None:
    load_environment()

    client = OpenAI()

    # Step 1:
    # Give the model the user request and the available tool definition.
    first_response = client.responses.create(
        model=MODEL,
        input=USER_REQUEST,
        tools=TOOLS,
    )

    print("\nFirst response output items:")

    for item in first_response.output:
        print(f"- {item.type}")

    # Step 2:
    # Find and execute the function call requested by the model.
    tool_outputs = []

    for item in first_response.output:
        if item.type != "function_call":
            continue

        arguments = json.loads(item.arguments)

        print("\nFunction requested by model")
        print(f"Name      : {item.name}")
        print(f"Arguments : {arguments}")

        result = execute_function_call(
            function_name=item.name,
            arguments=arguments,
        )

        print(f"Result    : {result}")

        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result),
            }
        )

    if not tool_outputs:
        print("\nThe model did not request a function call.")
        print(first_response.output_text)
        return

    # Step 3:
    # Send the function result back to the model.
    final_response = client.responses.create(
        model=MODEL,
        previous_response_id=first_response.id,
        input=tool_outputs,
        tools=TOOLS,
    )

    print("\n========================================")
    print("Final response")
    print("========================================")
    print(final_response.output_text)


if __name__ == "__main__":
    main()
