import json
import os
import time
import chromadb
from chromadb.config import Settings
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), verbose=True) #读取本地的.env文件,里面定义了OPENAI_API_KEY等环墶变量

'''
创建一个 OpenAI 客户端实例，用于调用 OpenAI 的 API。
这里假设 OpenAI 是一个已经导入的类（可能来自 OpenAI 官方 SDK），它会自动读取环境变量中的 OPENAI_API_KEY 来进行身份验证
'''
client = OpenAI()

#基于关键字检索的排序
class MyEsConnector:
    def __init__(self, es_client, index_name, keyword_fn):
        self.es_client = es_client
        self.index_name = index_name
        self.keyword_fn = keyword_fn
    #文档灌库,将文档提取关键字后存储到es数据库中
    def add_documents(self, documents):
        '''文档灌库'''
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)
        self.es_client.indices.create(index=self.index_name)
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "keywords": self.keyword_fn(doc),
                    "text": doc,
                    "id": f"doc_{i}"
                }
            }
            for i, doc in enumerate(documents)
        ]
        helpers.bulk(self.es_client, actions)
        time.sleep(1)

    #检索，将用户输入的文本转换为关键字，然后在es数据库中检索，返回检索结果
    def search(self, query_string, top_n=3):
        '''检索'''
        search_query = {
            "match":{
                "keywords": self.keyword_fn(query_string)
            }
        }
        res = self.es_client.search(
            index=self.index_name, query=search_query, size=top_n)
        return {
            hit["_source"]["id"]: {
                "text": hit["_source"]["text"],
                "rank": i,
            }
            for i, hit in enumerate(res["hits"]["hits"])
        }

from chinese_utils import to_keywords #使用中文的关键字提取函数
#引入配置文件
ELASTICSEARCH_BASE_URL = os.getenv("ELASTICSEARCH_BASE_URL")
ELASTICSEARCH_NAME = os.getenv("ELASTICSEARCH_NAME")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")

#基于文本向量检索的排序->语义相似度检索
class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        #内存模式
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        #数据持久化
        #chroma_client = chromadb.PersistentClient(path='./chroma')

        #注意:为了演示，实际不需要每次reset()，并且是不可逆的
        chroma_client.reset()

        #创建一个collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_document(self, documents):
        '''
        向collection中添加文档与向量
        '''
        self.collection.add(
            embeddings=self.embedding_fn(documents),#每个文档的向量
            documents=documents, #文档的原文
            ids=[f"id{i}" for i in range(len(documents))] #文档的id
        )

    def search(self, query, top_n):
        '''
        检索向量数据库
        '''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),#查询向量
            n_results=top_n #返回的最相似的文档数
        )
        return results

'''
将文本转换为向量
'''
def get_embeddings(texts, model='text-embedding-ada-002', dimensions=None):
    '''
    封装OpenAI的Embedding模型接口
    '''
    if model == 'text-embedding-ada-002':
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data #调用OpenAI的Embedding模型接口
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]  # 返回文本的向量

#基于RRF的融合排序
def rrf(ranks, k=1):
    ret = {}
    #遍历每次的排序结果
    for rank in ranks:
        #遍历排序中每个元素
        for id, val in rank.items():
            if id not in ret:
                ret[id] = {"score":0, "text":val["text"]}
            #计算RRF得分
            ret[id]["score"] += 1.0/(k+val["rank"])
     #按RRF得分排序，并返回
    return dict(sorted(ret.items(), key=lambda item:item[1]["score"], reverse=True))

# 背景说明：在医学中“小细胞肺癌”和“非小细胞肺癌”是两种不同的癌症
query = "非小细胞肺癌的患者"
documents = [
    "玛丽患有肺癌，癌细胞已转移",
    "刘某肺癌I期",
    "张某经诊断为非小细胞肺癌III期",
    "小细胞肺癌是肺癌的一种"
]

if __name__ == '__main__':
    es = Elasticsearch(
        hosts=[ELASTICSEARCH_BASE_URL],
        basic_auth=(ELASTICSEARCH_NAME, ELASTICSEARCH_PASSWORD),  # 用户名和密码
    )

    # 创建一个Elasticsearch连接器
    es_connector = MyEsConnector(es, "demo_es_rrf", to_keywords)
    # 文档灌库
    es_connector.add_documents(documents)
    # 关键字检索
    keyword_search_results = es_connector.search(query, 3)
    print("keyword_search_results:")
    print(json.dumps(keyword_search_results, indent=4, ensure_ascii=False))
    print("keyword_search_results end")

    # 创建一个向量数据库连接器
    vector_db_connector = MyVectorDBConnector("demo_vector_rrf", get_embeddings)
    #文档灌库
    vector_db_connector.add_document(documents)
    #向量检索
    vector_search_results = {
        "doc_"+str(documents.index(doc)):{
            "text":doc,
            "rank":i
        }
        for i, doc in enumerate(
            vector_db_connector.search(query, 3)["documents"][0]
        )
    } #把结果转成跟上面关键字检索结果一样的格式
    print("vector_search_results:")
    print(json.dumps(vector_search_results, indent=4, ensure_ascii=False))

    #结果对比 基于RRF的融合排序：关键字检索结果和向量检索结果的融合排序
    reranked = rrf([keyword_search_results, vector_search_results])
    print("reranked:")
    print(json.dumps(reranked, indent=4, ensure_ascii=False))