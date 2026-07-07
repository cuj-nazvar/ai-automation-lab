from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os

import requests

# Now we are making a pure REST API request to the OpenAI API WITHOUT using the OpenAI Python client library.
os.getenv("OPENAI_API_KEY")  # Ensure the API key is loaded from the .env file

# Find the repository root (two levels above this file)
repo_root = Path(__file__).resolve().parents[2]
load_dotenv(repo_root / ".env")

api_key = os.getenv("OPENAI_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "gpt-5.5",
    "input": "Explain Retrieval Augmented Generation in one sentence.",
}

response = requests.post(
    "https://api.openai.com/v1/responses", headers=headers, json=payload
)

response_json = response.json()
usage = response_json["usage"]
output_text = response_json["output"][0]["content"][0]["text"]

# Print essential information about the response in a human-readable format
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

# This is going to be a requests.models.Response
print(f"Type of response: {type(response)}")
# This is going to be a dict
print(f"Type of response.json(): {type(response.json())}")
# This is going to be a bytes
print(f"Type of response.content: {type(response.content)}")

# client = OpenAI()
#
# response = client.responses.create(
#    model="gpt-5.5",
#    input="Explain Retrieval Augmented Generation in 5 bullet points for a technical program manager.",
# )
#
# print(response.output_text)
