import json
from pathlib import Path
from urllib import response

from dotenv import load_dotenv
from openai import OpenAI


def load_environment() -> None:
    """Load the repository-level environment file."""

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


## This function returns a hardcoded dictionary representing the current project delivery status, including overall status, target date, and lists of open high and medium risks.
## This information can be retrieved from other subsystems or databases in a real-world scenario, but for this example, it is hardcoded for simplicity.
def get_project_status() -> dict:
    """Return the current project delivery status."""

    return {
        "overall_status": "AMBER",
        "target_date": "September 30",
        "open_high_risks": [
            "RISK-004",
        ],
        "open_medium_risks": [
            "RISK-002",
            "RISK-003",
        ],
    }


## This function takes a risk ID as input and returns the details of that specific project risk from a predefined dictionary of risks. If the risk ID is not found, it returns an error message indicating that the risk is unknown.
## In a real-world scenario, this information could be retrieved from a database or other data source, but for this example, it is hardcoded for simplicity.
def get_risk_details(risk_id: str) -> dict:
    """Return details for a specific project risk."""

    risks = {
        "RISK-002": {
            "title": "Certificate Provisioning",
            "severity": "MEDIUM",
            "status": "OPEN",
            "description": (
                "Manufacturing certificate provisioning integration is incomplete."
            ),
        },
        "RISK-003": {
            "title": "Production Monitoring",
            "severity": "MEDIUM",
            "status": "MITIGATING",
            "description": (
                "Application-level production alerts are not yet complete."
            ),
        },
        "RISK-004": {
            "title": "Load Test Environment",
            "severity": "HIGH",
            "status": "OPEN",
            "description": (
                "The current test environment cannot "
                "reliably generate the required peak load."
            ),
        },
    }

    return risks.get(
        risk_id,
        {
            "error": f"Unknown risk: {risk_id}",
        },
    )


TOOLS = [
    {
        "type": "function",
        "name": "get_project_status",
        "description": (
            "Get the current overall project delivery status "
            "and the IDs of open project risks."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_risk_details",
        "description": ("Get detailed information about a specific project risk."),
        "parameters": {
            "type": "object",
            "properties": {
                "risk_id": {
                    "type": "string",
                    "description": (
                        "The project risk identifier, for example RISK-004."
                    ),
                }
            },
            "required": ["risk_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


## The dispatcher
## More information about this can be found in 04-function-calling
def execute_function_call(
    function_name: str,
    arguments: dict,
) -> dict:
    """Execute a tool requested by the model."""

    if function_name == "get_project_status":
        return get_project_status()

    if function_name == "get_risk_details":
        return get_risk_details(**arguments)

    raise ValueError(f"Unknown function requested: {function_name}")


USER_REQUEST = """
Review the current project status.

Identify what is currently threatening delivery and recommend
which issue should be escalated first.

Use the available project tools to investigate before answering.
"""


def main() -> None:
    load_environment()

    client = OpenAI()

    response = client.responses.create(
        model="gpt-5.5",
        input=USER_REQUEST,
        tools=TOOLS,
    )

    ## The agent will continue to call tools until it has enough information to provide a final answer.
    # The loop checks for any function calls in the model's response and executes them,
    # feeding the results back into the model until no more function calls are present.
    while True:
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if not function_calls:
            print("\n========================================")
            print("FINAL ANSWER")
            print("========================================")
            print(response.output_text)
            break

        tool_outputs = []

        for function_call in function_calls:
            arguments = json.loads(function_call.arguments)

            print("\n========================================")
            print("ACTION")
            print("========================================")
            print(f"Tool      : {function_call.name}")
            print(f"Arguments : {arguments}")

            result = execute_function_call(
                function_call.name,
                arguments,
            )

            print("\nOBSERVATION")
            print(json.dumps(result, indent=2))

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
