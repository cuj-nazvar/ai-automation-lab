import tiktoken

MODEL = "gpt-4.1-mini"  # or another compatible model

text = "Hello GPT! I love learning about how LLMs work."

encoding = tiktoken.encoding_for_model(MODEL)

tokens = encoding.encode(text)

print(text)
print()
print(tokens)
print()
print(f"Number of tokens: {len(tokens)}")
print("Decoding tokens back to text")
for token in tokens:
    print(token, "->", encoding.decode([token]))
