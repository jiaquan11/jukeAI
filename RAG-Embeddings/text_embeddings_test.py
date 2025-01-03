import numpy as np
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
from pyexpat.errors import messages

_ = load_dotenv(find_dotenv(), verbose=True) #读取本地的.env文件,里面定义了OPENAI_API_KEY等环墶变量

'''
创建一个 OpenAI 客户端实例，用于调用 OpenAI 的 API。
这里假设 OpenAI 是一个已经导入的类（可能来自 OpenAI 官方 SDK），它会自动读取环境变量中的 OPENAI_API_KEY 来进行身份验证
'''
client = OpenAI()

def cos_sim(a, b):
    '''
    余弦距离---越大越相似
    '''
    return dot(a, b) / (norm(a) * norm(b))

def l2(a, b):
    '''
    欧式距离---越小越相似
    '''
    x = np.asarray(a) - np.asarray(b) # 将向量转换为数组,然后相减,得到一个新的数组,即向量的差值
    return norm(x)#求向量差值的模

model = "text-embedding-3-large"
dimensions = 128

'''
将文本转换为向量
'''
def get_embeddings(texts, model='text-embedding-ada-002', dimensions = None):
    '''
    封装OpenAI的Embedding模型接口
    '''
    if model == 'text-embedding-ada-002':
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data] #返回文本的向量

if __name__ == '__main__':
    #测试
    #test_query = ['测试文本']
    #vec = get_embeddings(test_query)[0] #获取测试文本的向量
    #print(f"Total dimensions: {len(vec)}") #输出向量的维度
    #print(f"First 10 dimensions: {vec[:10]}") #输出向量的前10个维度

    #query = "国际争端"
    # 且能支持跨语言
    query = "global conflicts"
    documents = [
        "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
        "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
        "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
        "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
        "我国首次在空间站开展舱外辐射生物学暴露实验",
    ]

    query_vec = get_embeddings([query], model=model,dimensions=dimensions)[0]
    doc_vecs = get_embeddings(documents, model=model,dimensions=dimensions)
    print('向量维度:{}'.format(len(query_vec)))
    print()

    print("Query与自己的余弦距离: {:.2f}".format(cos_sim(query_vec, query_vec)))
    print("Query与Documents的余弦距离:")
    for vec in doc_vecs:
        print(cos_sim(query_vec, vec))

    print() #空行

    print("Query与自己的欧氏距离: {:.2f}".format(l2(query_vec, query_vec)))
    print("Query与Documents的欧氏距离:")
    for vec in doc_vecs:
        print(l2(query_vec, vec))
