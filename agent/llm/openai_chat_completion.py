from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8743/v1"

# completion = client.completions.create(model="meta-llama/Llama-3.3-70B-Instruct",
#                                       prompt="San Francisco is a")
# print("Completion result:", completion.choices[0].text)

def get_response(prompt):
    pass
    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,)

    completion = client.completions.create(model="meta-llama/Llama-3.3-70B-Instruct",prompt=prompt)

    return completion.choices[0].text
