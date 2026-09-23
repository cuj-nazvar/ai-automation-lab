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

## This is the generic dispatcher, which receives a Python function name and a dictionary of arguments, and routes the call to the appropriate function. This is the mechanism that allows the model to call Python functions.
##
## Essentially the dispatching will look like e.g:
##                  LLM
##                   │
##                   │ requests:
##                   │
##                   │ "calculate_project_capacity"
##                   │ + arguments
##                   ▼
##         execute_function_call()
##              dispatcher
##                   │
##        ┌──────────┼──────────┐
##        ▼          ▼          ▼
## token_volume   token_cost   project_capacity


def execute_function_call(function_name: str, arguments: dict) -> dict:
    """Route a function call requested by the model to the Python implementation."""

    if function_name == "calculate_token_volume":
        return calculate_token_volume(**arguments)

    if function_name == "calculate_token_cost":
        return calculate_token_cost(**arguments)

    if function_name == "calculate_project_capacity":
        return calculate_project_capacity(**arguments)

    # This is already a basig guardrail. It will not be reached if the model is using the tools parameter correctly, but it is a good safety check.
    raise ValueError(f"Unknown function requested: {function_name}")


def load_environment() -> None:
    """Load the repository-level environment file."""

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


'''
USER_REQUEST = """
We have a project with approximately 650 hours of work remaining.

There are 4 engineers available, and each engineer can dedicate
about 25 hours per week to this project.

How many weeks should we expect the remaining work to take?
"""

The USER_REQUEST above yields the following result:

Response output items:
- function_call

Function requested by model
Name      : calculate_project_capacity
Arguments : {'total_work_hours': 650, 'number_of_people': 4, 'hours_per_person_per_week': 25}
Result    : {'total_work_hours': 650, 'weekly_capacity': 100, 'required_weeks': 6.5}

USER_REQUEST = """
We expect 1,500 API requests.
Each request contains approximately 4,000 input tokens
and generates around 600 output tokens.

How many tokens will we process in total?
"""

The USER_REQUEST above yields the following result:
Response output items:
- function_call

Function requested by model
Name      : calculate_token_volume
Arguments : {'number_of_requests': 1500, 'input_tokens_per_request': 4000, 'output_tokens_per_request': 600}
Result    : {'number_of_requests': 1500, 'total_input_tokens': 6000000, 'total_output_tokens': 900000, 'total_tokens': 6900000}

USER_REQUEST = """
We expect to process 2,400,000 input tokens and
350,000 output tokens.

Input tokens cost 2.50 per million and output tokens
cost 10.00 per million.

How much will that cost?
"""

The USER_REQUEST above yields the following result:
Response output items:
- function_call

Function requested by model
Name      : calculate_token_cost
Arguments : {'input_tokens': 2400000, 'output_tokens': 350000, 'input_price_per_million': 2.5, 'output_price_per_million': 10}
Result    : {'input_cost': 6.0, 'output_cost': 3.5, 'total_cost': 9.5}


USER_REQUEST = """
We are planning 200 API calls per hour per engineer, and have a team of 4 engineers for the remaining 5 weeks.

Each engineer can dedicate 20 hours per week to this project.

Our APIs are expected to process 3,000 input tokens and 500 output tokens per request. Input tokens cost 2.50 per million and output tokens cost 10.00 per million.

How much it will cost to process all the requests in the remaining 5 weeks?
"""

The USER REQUEST above yields the following result:

Response output items:
- reasoning
- function_call

Function requested by model
Name      : calculate_token_volume
Arguments : {'number_of_requests': 80000, 'input_tokens_per_request': 3000, 'output_tokens_per_request': 500}
Result    : {'number_of_requests': 80000, 'total_input_tokens': 240000000, 'total_output_tokens': 40000000, 'total_tokens': 280000000}
- function_call

Function requested by model
Name      : calculate_token_cost
Arguments : {'input_tokens': 240000000, 'output_tokens': 40000000, 'input_price_per_million': 2.5, 'output_price_per_million': 10}
Result    : {'input_cost': 600.0, 'output_cost': 400.0, 'total_cost': 1000.0}

Essentially, this is what is happening, i.e. the model forms both calls in parallel, and not sequentially, and then the Python code executes both afterwards.
This is NOT what we want to have in the real world, but it is a good demonstration of how the model can call multiple functions in parallel.

                    MODEL RESPONSE
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
calculate_token_volume()      calculate_token_cost()
      80,000                       240M / 40M
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  Python executes
                   both afterwards
'''

USER_REQUEST = """
First calculate the total token volume for 80,000 API requests.

Each request contains 3,000 input tokens and 500 output tokens.

After the token volume has been calculated, use the ACTUAL RESULT returned
by the calculate_token_volume tool to calculate the cost.

Input tokens cost 2.50 per million and output tokens cost 10.00 per million.

Do not calculate or infer the token totals yourself.
You must wait for the result of calculate_token_volume before calling
calculate_token_cost.

What is the total cost?
"""


def main() -> None:
    load_environment()

    client = OpenAI()

    response = client.responses.create(
        model="gpt-5.5",
        input=USER_REQUEST,
        tools=TOOLS,
    )

    while True:
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        # No more function calls means the model is finished.
        if not function_calls:
            print("\n========================================")
            print("Final response")
            print("========================================")
            print(response.output_text)
            break

        tool_outputs = []

        for function_call in function_calls:
            arguments = json.loads(function_call.arguments)

            print("\nFunction requested by model")
            print(f"Name      : {function_call.name}")
            print(f"Arguments : {arguments}")

            result = execute_function_call(
                function_name=function_call.name,
                arguments=arguments,
            )

            print(f"Result    : {result}")

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": json.dumps(result),
                }
            )

        response = client.responses.create(
            model="gpt-5.5",
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOLS,
        )


if __name__ == "__main__":
    main()
