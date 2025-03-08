import requests
import json
import os

url = "https://api.openai-hk.com/v1/chat/completions"

payload = json.dumps({
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "system",
            "content": "assistant"
        },
        {
            "role": "user",
            "content": "你叫什么名字?"
        }
    ]
})

headers = {
    'Accept': 'application/json',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + os.getenv('OPENAI_API_KEY'),
    'Host': 'api.openai-hk.com',
    'Connection': 'keep-alive'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
