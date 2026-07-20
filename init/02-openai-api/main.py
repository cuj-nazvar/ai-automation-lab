from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os
import requests

API_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-5.5"
PROMPT = "Explain Retrieval Augmented Generation in one sentence."


def main() -> None:

    load_environment()
    api_key = get_openai_api_key()

    print("\nRunning raw HTTPS request to OpenAI API...")
    run_raw_https_request(api_key)
    print("\nRunning OpenAI SDK request to OpenAI API...")
    run_openai_sdk_request()


def load_environment() -> None:
    # Load environment variables from the repository-level .env file.
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".env"
    if not env_file.exists():
        raise FileNotFoundError(f".env file not found at {env_file}")

    load_dotenv(env_file)


def get_openai_api_key() -> str:
    # Return the OpenAI API key or fail with a clear error.
    open_api_key = os.getenv("OPENAI_API_KEY")
    if not open_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    return open_api_key


def run_raw_https_request(open_api_key: str) -> None:
    # This is the very basic way to make a request to the OpenAI API. It is useful for understanding how the API works and for debugging issues with the OpenAI Python client library.
    headers = {
        "Authorization": f"Bearer {open_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": PROMPT,
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=10,  # seconds
        )
    except requests.RequestException as e:
        print(f"Error making request: {e}")
        return

    if response.ok:
        print_response_summary(response)

        # This is going to be a requests.models.Response
        print(f"Type of response: {type(response)}")
        # This is going to be a dict
        print(f"Type of response.json(): {type(response.json())}")
        # This is going to be a bytes
        print(f"Type of response.content: {type(response.content)}")
    else:
        print_error_summary(response)


def run_openai_sdk_request() -> None:
    # This is the recommended way to make a request to the OpenAI API. It is easier to use and provides better error handling.
    client = OpenAI()
    try:
        response = client.responses.create(
            model=MODEL,
            input=PROMPT,
        )
    except Exception as e:
        print(f"Error making request: {e}")
        return

    print_sdk_response_summary(response)


def print_response_summary(response):
    response_json = response.json()
    usage = response_json["usage"]
    output_text = response_json["output"][0]["content"][0]["text"]

    print(f"""
    ========================================
    OpenAI Response Summary
    ========================================

    HTTP Status    : {response.status_code} {response.reason}

    Model          : {response_json["model"]}
    Response ID    : {response_json["id"]}
    Created At     : {response_json["created_at"]}

    Input Tokens   : {usage["input_tokens"]}
    Output Tokens  : {usage["output_tokens"]}
    Total Tokens   : {usage["total_tokens"]}

    Response Length: {len(response.text)} characters
    Output Text    : {output_text}

    ========================================
    """)


def print_sdk_response_summary(sdk_response) -> None:
    usage = sdk_response.usage

    print(
        f"""
    ========================================
    OpenAI SDK Response Summary
    ========================================

    Model           : {sdk_response.model}
    Response ID     : {sdk_response.id}
    Created At      : {sdk_response.created_at}

    Input Tokens    : {usage.input_tokens}
    Output Tokens   : {usage.output_tokens}
    Total Tokens    : {usage.total_tokens}

    Output Text     : {sdk_response.output_text}

    ========================================
    """
    )


def print_error_summary(response):
    try:
        response_json = response.json()
    except ValueError:
        response_json = {}

    error = response_json.get("error", {})

    print(f"""
    ========================================
    OpenAI Error Summary
    ========================================

    HTTP Status : {response.status_code} {response.reason}

    Type        : {error.get("type", "No error type provided")}
    Code        : {error.get("code", "No error code provided")}
    Parameter   : {error.get("param", "No error parameter provided")}

    Message     : {error.get("message", "No error message provided")}

    ========================================
    """)


if __name__ == "__main__":
    main()
