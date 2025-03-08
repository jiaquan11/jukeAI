from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "你是一个助手，请用中文输出JSON"},
        {"role": "user", "content": "帮我写一个冒泡算法?"}
    ]
)
print(response.choices[0].message.content)
