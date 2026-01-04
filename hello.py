from openai import OpenAI

key = r"sk-LhKF1lpfUxLjZegVA8ZAgA"

client = OpenAI(api_key=key, base_url=r'https://llmapi.blsc.cn/v1')

response = client.chat.completions.create(
    model='DeepSeek-V3.1',
    messages=[
        {'role': 'user', 'content': 'who are you?'}
    ],
    temperature=0.5,
    max_tokens=128,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
).choices[0].message.content

print(response)