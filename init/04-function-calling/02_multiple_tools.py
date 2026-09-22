import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-5.5"


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


def calculate_token_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict:
    """Calculate the cost of processing input and output tokens."""

    input_cost = input_tokens / 1_000_000 * input_price_per_million

    output_cost = output_tokens / 1_000_000 * output_price_per_million

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calculate_project_capacity(
    total_work_hours: float,
    number_of_people: int,
    hours_per_person_per_week: float,
) -> dict:
    """Calculate how many weeks are required to complete a workload."""

    weekly_capacity = number_of_people * hours_per_person_per_week

    required_weeks = total_work_hours / weekly_capacity

    return {
        "total_work_hours": total_work_hours,
        "weekly_capacity": weekly_capacity,
        "required_weeks": required_weeks,
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
                    "description": "Average input tokens per API request.",
                },
                "output_tokens_per_request": {
                    "type": "integer",
                    "description": "Average output tokens per API request.",
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
    },
    {
        "type": "function",
        "name": "calculate_token_cost",
        "description": (
            "Calculate the cost of processing a known number of "
            "input and output tokens."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "input_tokens": {
                    "type": "integer",
                    "description": "Total number of input tokens.",
                },
                "output_tokens": {
                    "type": "integer",
                    "description": "Total number of output tokens.",
                },
                "input_price_per_million": {
                    "type": "number",
                    "description": "Price per one million input tokens.",
                },
                "output_price_per_million": {
                    "type": "number",
                    "description": "Price per one million output tokens.",
                },
            },
            "required": [
                "input_tokens",
                "output_tokens",
                "input_price_per_million",
                "output_price_per_million",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "calculate_project_capacity",
        "description": (
            "Calculate how many weeks are required to complete a workload "
            "given the number of people and their weekly availability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "total_work_hours": {
                    "type": "number",
                    "description": "Total estimated work in hours.",
                },
                "number_of_people": {
                    "type": "integer",
                    "description": "Number of people available.",
                },
                "hours_per_person_per_week": {
                    "type": "number",
                    "description": (
                        "Number of hours each person can contribute per week."
                    ),
                },
            },
            "required": [
                "total_work_hours",
                "number_of_people",
                "hours_per_person_per_week",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def execute_function_call(function_name: str, arguments: dict) -> dict:
    """Route a function call requested by the model to the Python implementation."""

    if function_name == "calculate_token_volume":
        return calculate_token_volume(**arguments)

    if function_name == "calculate_token_cost":
        return calculate_token_cost(**arguments)

    if function_name == "calculate_project_capacity":
        return calculate_project_capacity(**arguments)

    raise ValueError(f"Unknown function requested: {function_name}")


def load_environment() -> None:
    """Load the repository-level environment file."""

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


USER_REQUEST = """
We have a project with approximately 650 hours of work remaining.

There are 4 engineers available, and each engineer can dedicate
about 25 hours per week to this project.

How many weeks should we expect the remaining work to take?
"""


def main() -> None:
    load_environment()

    client = OpenAI()

    response = client.responses.create(
        model="gpt-5.5",
        input=USER_REQUEST,
        tools=TOOLS,
    )

    print("\nResponse output items:")

    for item in response.output:
        print(f"- {item.type}")

        if item.type == "function_call":
            arguments = json.loads(item.arguments)

            print("\nFunction requested by model")
            print(f"Name      : {item.name}")
            print(f"Arguments : {arguments}")


if __name__ == "__main__":
    main()
