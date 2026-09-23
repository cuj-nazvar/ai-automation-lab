# Function Calling — Learning Notes

## 1. Function Calling

### Definition / Explanation

Function calling allows an LLM to request that an application execute a predefined external capability using structured arguments.

The model does not execute the Python function itself. It selects a tool and produces arguments matching the tool's schema. The application executes the actual function and can return the result to the model.

### Key Characteristics

- The model sees the tool definition, not the Python implementation.
- A tool definition contains a name, description, and parameter schema.
- `tools=TOOLS` exposes available capabilities to the model.
- The model maps natural language into structured function arguments.
- Python remains responsible for deterministic execution.
- Tool results can be returned to the model for further reasoning or a final answer.


## 2. Multiple Tools

### Definition / Explanation

Multiple tools allow the model to choose between several capabilities based on the user's intent.

The model uses the tool names, descriptions, and parameter schemas to decide which capability is appropriate.

### Key Characteristics

- Providing multiple tools does not mean all tools will be used.
- The model can select one or several tools from the available set.
- Multiple independent function calls can be returned in a single model response.
- Tool descriptions and schemas strongly influence tool selection.
- A dispatcher maps model-requested tool names to approved Python implementations.


## 3. Parallel vs. Sequential Tool Calls

### Definition / Explanation

Multiple function calls in one model response are not necessarily sequential.

If the model can infer all required arguments from the original request, it may request several tools at once.

Sequential tool calling is required when a later tool depends on the actual result returned by an earlier tool.

### Key Characteristics

Parallel / independent:

User request
→ Model
→ Tool A call + Tool B call
→ Application executes both

Sequential / dependent:

User request
→ Model
→ Tool A
→ Application executes Tool A
→ Tool A result returned to model
→ Model decides to call Tool B
→ Application executes Tool B
→ Tool B result returned to model
→ Final response


## 4. The Tool-Calling Loop

### Definition / Explanation

A tool-calling loop allows the model to repeatedly choose actions, observe their results, and decide what to do next until no further tool calls are required.

### Key Characteristics

The basic loop is:

1. Send the user request to the model.
2. Inspect the model response for function calls.
3. Execute requested functions.
4. Return the function results to the model.
5. Repeat until the model requests no more functions.
6. Return the model's final natural-language response.

Conceptually:

Model decision
→ Action
→ Observation
→ Model decision
→ Action
→ Observation
→ Final answer

This loop is an important building block for agents.


## 5. Application Control Boundary

### Definition / Explanation

The LLM can request actions, but the application decides which actions are actually executable.

The dispatcher therefore acts as a control boundary between model-generated requests and application code.

### Key Characteristics

- The model cannot directly execute arbitrary Python functions.
- Only explicitly exposed and mapped functions can be executed.
- Unknown function names can be rejected.
- Argument validation, permissions, iteration limits, and other guardrails can be added around this boundary.