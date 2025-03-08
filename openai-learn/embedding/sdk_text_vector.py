from openai import OpenAI

# 初始化 OpenAI 服务。
client = OpenAI()

# 调用嵌入 API
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# 示例文本
text = "Hello, world!"

# 获取嵌入向量
embedding = get_embedding(text)
print("Embedding vector:", embedding)
