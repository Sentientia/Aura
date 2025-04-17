from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8743/v1"

# completion = client.completions.create(model="meta-llama/Llama-3.3-70B-Instruct",
#                                       prompt="San Francisco is a")
# print("Completion result:", completion.choices[0].text)

def get_response(messages,model="meta-llama/Llama-3.3-70B-Instruct"):
    pass
    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,)

    completion = client.chat.completions.create(model=model,messages=messages, temperature=0.1, max_tokens=2000)

    return completion.choices[0].message.content

def get_history_as_strings(history):
    output_string = ""

    for message in history:
        role = message["role"]
        content = message["content"]
        output_string += f"{role}: {content}\n"
    
    return output_string

