import requests
import json
import os

url = os.getenv('OPENAI_BASE_URL') + "/embeddings"

payload = json.dumps({
    "model": "text-embedding-ada-002",
    "input": "cat"
})
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + os.getenv('OPENAI_API_KEY'),
}
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)
