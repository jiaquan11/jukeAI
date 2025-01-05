import requests

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"

#不使用Token进行匿名访问
response = requests.post(API_URL, json={'inputs': '您好，Hugging face'})
print(response.json())