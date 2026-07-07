# 02 - OpenAI API

## Objective

Understand how Python communicates with an LLM.

## Concepts

- SDK
- REST API
- HTTPS
- Authentication
- API Keys
- Environment Variables
- JSON serialization
- JSON deserialization

## Architecture Diagram

┌──────────────────────────────┐
│ Python Application           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ OpenAI Python SDK            │    
│ - JSON serialization         │
│ - HTTP client                │
│ - Authentication headers     │
│ - Retry logic                │
└──────────────┬───────────────┘
               │ HTTPS
               ▼
┌──────────────────────────────┐
│ OpenAI Platform              │
│ - Authenticate API key       │
│ - Authorize request          │
│ - Validate request           │
│ - Select model               │
│ - Execute inference          │
│ - Track usage & billing      │
└──────────────┬───────────────┘
               ▼
        GPT-5.5 Model


## Model Control Parameters

The output of the request contains many parameters, which are good to understand. Some expamples below:

| Field              | Controls                            | Think of it as...          |
| ------------------ | ----------------------------------- | -------------------------- |
| `background`       | **Execution mode**                  | Sync vs async processing   |
| `reasoning.effort` | **How hard the model thinks**       | CPU time / thinking budget |
| `temperature`      | **How deterministic the output is** | Randomness / creativity    |

Interestingly, these control parameters are always returned in the request's response. Nevertheless, it's worth categorizing these, and there would be 3 categories:
1. Request Configuration -> Things that control execution.
** model
** temperature
** reasoning
** background
** max_output_tokens

2. Execution Metadata -> Things created by the platform.
** id
** created_at
** status

3. Execution Results -> Things only known afterwards.
** usage
** output
** finish_reason

## What surprised me?

(To be filled after implementation)

## Possible enterprise applications

(To be filled after implementation)

## Questions for Further Investigation

- Why does the platform return reasoning?
- How does tokenization work?
- How is cost calculated?
- What happens if the request fails?
- Why does the SDK exist?