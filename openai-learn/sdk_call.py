from openai import OpenAI
# pip install openai==1.40.3
import os
# 从环境变量中读取OPENAI_BASE_URL
print(os.getenv('OPENAI_BASE_URL'))
# 初始化 OpenAI 服务。
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "assistant"},
        {"role": "user", "content": "Hello"}
    ]
)
print(completion.choices[0].message.content)