import json
from pathlib import Path
from urllib import response

from dotenv import load_dotenv
from openai import OpenAI


def load_environment() -> None:
    """Load the repository-level environment file."""

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


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


def get_risk_details(risk_id: str) -> dict:
    """Return details for a specific project risk."""

    risks = {
        "RISK-002": {
            "title": "Certificate Provisioning",
            "severity": "MEDIUM",
            "status": "OPEN",
            "estimated_delay_days": 5,
            "description": (
                "Manufacturing certificate provisioning integration is incomplete."
            ),
        },
        "RISK-003": {
            "title": "Production Monitoring",
            "severity": "MEDIUM",
            "status": "MITIGATING",
            "estimated_delay_days": 3,
            "description": (
                "Application-level production alerts are not yet complete."
            ),
        },
        "RISK-004": {
            "title": "Load Test Environment",
            "severity": "HIGH",
            "status": "OPEN",
            "estimated_delay_days": 12,
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


def get_risk_appetite() -> dict:
    """Return the organization's risk appetite and escalation rules."""

    return {
        "schedule_risk_appetite": "LOW",
        "technical_risk_appetite": "LOW",
        "cost_risk_appetite": "MEDIUM",
        "rules": {
            "maximum_accepted_schedule_impact_days": 3,
            "high_risks_require_escalation": True,
            "critical_risks_require_escalation": True,
        },
    }


## Deterministic function to calculate schedule impact based on estimated delay and available buffer
# This function is intentionally not AI, so that the agent can use it as a tool to calculate the schedule impact of a risk.
# It takes the estimated delay in days and the available buffer in days as input, and returns a dictionary containing the estimated delay, available buffer, calculated schedule impact, and whether the target date is at risk.
def calculate_schedule_impact(
    estimated_delay_days: int,
    available_buffer_days: int,
) -> dict:
    """Calculate the expected schedule impact of a delay."""

    schedule_impact_days = max(
        0,
        estimated_delay_days - available_buffer_days,
    )

    return {
        "estimated_delay_days": estimated_delay_days,
        "available_buffer_days": available_buffer_days,
        "schedule_impact_days": schedule_impact_days,
        "target_date_at_risk": schedule_impact_days > 0,
    }


ESCALATIONS = []


def create_escalation(
    risk_id: str,
    reason: str,
    priority: str,
    proposed_treatment: str,
) -> dict:
    """Create a project escalation with a proposed risk treatment."""

    escalation = {
        "risk_id": risk_id,
        "reason": reason,
        "priority": priority,
        "proposed_treatment": proposed_treatment,
        "status": "CREATED",
    }

    ESCALATIONS.append(escalation)

    return escalation


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
    {
        "type": "function",
        "name": "calculate_schedule_impact",
        "description": (
            "Calculate whether an estimated delay exceeds "
            "the available schedule buffer and threatens "
            "the target date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "estimated_delay_days": {
                    "type": "integer",
                    "description": ("Estimated number of days of delay."),
                },
                "available_buffer_days": {
                    "type": "integer",
                    "description": ("Number of schedule buffer days available."),
                },
            },
            "required": [
                "estimated_delay_days",
                "available_buffer_days",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "create_escalation",
        "description": (
            "Create a formal project escalation for a risk "
            "that requires management attention, including "
            "a proposed risk treatment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "risk_id": {
                    "type": "string",
                },
                "reason": {
                    "type": "string",
                },
                "priority": {
                    "type": "string",
                    "enum": [
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "CRITICAL",
                    ],
                },
                "proposed_treatment": {
                    "type": "string",
                    "enum": [
                        "MITIGATE",
                        "AVOID",
                        "ACCEPT",
                        "TRANSFER",
                    ],
                    "description": (
                        "Recommended treatment strategy for the project risk."
                    ),
                },
            },
            "required": [
                "risk_id",
                "reason",
                "priority",
                "proposed_treatment",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_risk_appetite",
        "description": (
            "Get the organization's risk appetite and rules "
            "for deciding whether project risks are acceptable "
            "or require treatment and escalation."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def execute_function_call(
    function_name: str,
    arguments: dict,
) -> dict:
    """Execute a tool requested by the model."""

    if function_name == "get_project_status":
        return get_project_status()

    if function_name == "get_risk_details":
        return get_risk_details(**arguments)

    if function_name == "calculate_schedule_impact":
        return calculate_schedule_impact(**arguments)

    if function_name == "create_escalation":
        return create_escalation(**arguments)

    if function_name == "get_risk_appetite":
        return get_risk_appetite()

    raise ValueError(f"Unknown function requested: {function_name}")


USER_REQUEST = """
Review the current project status and investigate the risks
that could threaten the September 30 target date.

There are 4 days of schedule buffer remaining.

Use the organization's risk appetite when deciding whether
the identified risks are acceptable.

Determine which risk poses the greatest schedule threat and
recommend an appropriate treatment strategy.

If you conclude that a risk requires management attention,
create an escalation with the proposed treatment.

Explain your final decision.
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

    print("\n========================================")
    print("ESCALATIONS CREATED")
    print("========================================")
    print(json.dumps(ESCALATIONS, indent=2))


if __name__ == "__main__":
    main()
