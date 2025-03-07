import requests

#这是一个续写生成模型，不是问答模型
API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

#使用Token进行访问在线模型
response = requests.post(API_URL, headers=headers, json={'inputs': '您好，Hugging face'})
print(response.json())