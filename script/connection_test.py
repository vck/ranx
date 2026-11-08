from openai import OpenAI

# Point the client to your local server's address and port
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local-bypass",  # A placeholder key is required but not used
)

response = client.chat.completions.create(
    model="local-model",  # Model name can be anything when using a local server
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
