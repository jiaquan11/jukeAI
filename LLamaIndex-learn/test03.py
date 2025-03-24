import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import ServiceContext

#定义向量存储数据库
chroma_client = chromadb.PersistentClient()
# #创建集合
# # chroma_client.create_collection("quickstart")
# # print("集合创建完毕")

# #获取已经存在的向量数据库
# chroma_collection = chroma_client.get_collection("quickstart")
# print(chroma_collection)
# print("获取已经存在的知识库")

#尝试获取集合，如果不存在则创建
try:
    chroma_collection = chroma_client.get_collection("quickstart")
    print("使用已经存在的本地知识库")
except chromadb.errors.InvalidCollectionException:
    chroma_client.create_collection("quickstart")
    print("创建一个全新的本地知识库")

#声明向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = ServiceContext.from_defaults(vector_store=vector_store)
