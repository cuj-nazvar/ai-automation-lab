# Agents — Learning Notes

## AI Agents

### Definition / Explanation

An AI agent is an LLM operating inside a control loop where it can
select actions, observe the results of those actions, and continue
working until it considers a goal complete.

The model itself is not the entire agent.

The agent is the larger system consisting of:

- an LLM
- a goal or task
- available tools
- an execution environment
- observations returned by the environment
- a control loop

A simplified agent loop is:

    Goal
      ↓
    Model
      ↓
    Action
      ↓
    Environment
      ↓
    Observation
      ↓
    Model
      ↓
     ...
      ↓
    Final Answer


## Function Calling vs Agents

Function calling allows an LLM to request that an application execute
a specific capability.

An agent builds on this mechanism by repeatedly allowing the model to
decide what to do next based on previous observations.

Function calling:

    User Request
        ↓
      Model
        ↓
    Tool Call
        ↓
    Tool Result
        ↓
    Final Answer

Agent:

    Goal
      ↓
    Model
      ↓
    Tool Call
      ↓
    Observation
      ↓
    Model
      ↓
    Another Tool Call
      ↓
    Observation
      ↓
     ...
      ↓
    Final Answer

The important difference is that the application does not prescribe
the complete sequence of actions in advance.

The model can dynamically decide which capability to use next based on
what it has learned so far.


## The Agent Loop

### Definition / Explanation

The agent loop repeatedly checks whether the model requested additional
tool calls.

If tools are requested:

1. The application executes the requested tools.
2. Their results are returned to the model as observations.
3. The model evaluates the new information.
4. The model decides whether another action is required.

If no further tool calls are requested, the loop terminates and the
model returns its final answer.

Conceptually:

    while goal_not_complete:

        model decides

            ↓

        action

            ↓

        tool execution

            ↓

        observation

            ↓

        model decides again


### Key Characteristics

- The user provides a goal rather than a fixed sequence of steps.
- The model selects tools dynamically.
- Tool results become observations for subsequent decisions.
- The model can perform multiple reasoning/action cycles.
- The execution path can differ depending on what previous tools return.
- The model determines when it has enough information to finish.


## Multi-Tool Agents

### Definition / Explanation

An agent can be given tools with different purposes rather than several
variations of the same capability.

In `02_multi_tool_agent.py`, the agent was given tools representing
different types of interaction with its environment:

    READ
    get_project_status()

    READ
    get_risk_details()

    READ / POLICY
    get_risk_appetite()

    COMPUTE
    calculate_schedule_impact()

    WRITE / ACTION
    create_escalation()

The agent can decide which of these capabilities are necessary for
achieving its goal and in which order they should be used.


### Key Characteristics

- Tool descriptions define the capabilities visible to the model.
- The model does not see the underlying Python implementation.
- Different tools can represent information retrieval, deterministic
  calculations, organizational policies, or external actions.
- The same user goal can result in different tool-call sequences.
- Multiple independent tool calls may also be requested in the same
  model response.


## Deterministic Tools

Not every operation should be performed through LLM reasoning.

Operations such as schedule calculations can be implemented as
deterministic tools:

    estimated delay
          +
    available buffer
          ↓
    schedule impact

This separates language-based reasoning from deterministic business
logic.

The LLM decides when the calculation is needed, while normal software
performs the calculation.


## Policy-Informed Decisions

A risk treatment is not determined only by the characteristics of the
risk itself.

The same schedule exposure may be acceptable to one organization and
unacceptable to another.

A decision can therefore depend on:

    Risk Information
          +
    Schedule Impact
          +
    Organizational Risk Appetite
          ↓
    Treatment Recommendation

In the exercise, changing the organization's maximum accepted schedule
impact could change the agent's recommendation without changing the
underlying project risk.

This demonstrates that useful agent decisions often require both
operational data and organizational context.


## Risk Treatment

The multi-tool agent can propose one of four standard treatment
strategies:

- MITIGATE — reduce the probability or impact of the risk
- AVOID — change the plan so that the risk is eliminated
- ACCEPT — consciously tolerate the remaining exposure
- TRANSFER — move responsibility or exposure to another party

The allowed values are constrained by the tool schema rather than left
as unrestricted model-generated text.

The agent selects a treatment based on the information it has gathered,
but the actual treatment decision can still be subject to human review.


## Read Tools vs Write Tools

A significant distinction exists between tools that observe the world
and tools that change it.

Read tools:

    get_project_status()
    get_risk_details()
    get_risk_appetite()

These retrieve information without changing external state.

Write tools:

    create_escalation()

These create side effects.

In a real system, a write tool could represent actions such as:

- creating a Jira issue
- sending an email
- modifying a project record
- approving a workflow
- changing a deployment
- committing money or resources

Giving an agent access to a write tool therefore introduces additional
risk compared with giving it read-only capabilities.


## Tool Schemas as Control Boundaries

Tool schemas constrain how the model can interact with application
capabilities.

For example, the escalation tool restricts the treatment strategy to:

    MITIGATE
    AVOID
    ACCEPT
    TRANSFER

This prevents arbitrary values from being supplied through that
parameter.

However, schema validation alone does not make an agent safe.

The application still controls:

- which tools are exposed
- which arguments are accepted
- whether actions require approval
- what permissions the tools have
- how many iterations are allowed
- what resources the agent can access


## Current Agent Limitations

The agents built so far intentionally have very few safeguards.

The current control loop can conceptually continue indefinitely:

    while True:
        ...

There is currently no:

- maximum iteration count
- execution timeout
- token or cost budget
- permission model
- human approval step
- protection against malicious instructions
- validation of whether an action is appropriate
- restriction on side-effecting operations

These limitations are intentional at this stage.

They provide concrete problems to address later when studying
guardrails and human-in-the-loop agent architectures.


## Main Lessons So Far

An agent is not simply an LLM with more intelligence.

It is a system architecture in which an LLM is allowed to interact with
an environment through controlled capabilities.

The core pattern is:

    Observe
      ↓
    Decide
      ↓
    Act
      ↓
    Observe
      ↓
    Decide again

Function calling provides the mechanism for taking actions.

The agent loop provides the mechanism for repeatedly deciding which
actions to take.

Tools provide access to external information and capabilities.

Deterministic software can handle calculations and business rules while
the LLM handles interpretation and decision-making.

Organizational context such as risk appetite can materially change an
agent's recommendations.

Finally, read access and write access are fundamentally different.
Once an agent can change external state, permissions, validation,
approval and other guardrails become part of the system design.