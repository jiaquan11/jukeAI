import chromadb
from chromadb.config import Settings
from openai import OpenAI

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

#加载环境变量
'''
使用 python-dotenv 库加载 .env 文件中的环境变量。
.env 文件通常用于存储敏感信息，例如 API 密钥（OPENAI_API_KEY）。
find_dotenv()：
自动查找项目中的 .env 文件路径。
load_dotenv()：
加载 .env 文件中的环境变量到系统环境中。
verbose=True：
如果设置为 True，会输出加载 .env 文件的详细信息，方便调试。
'''
from dotenv import load_dotenv, find_dotenv
from pyexpat.errors import messages

_ = load_dotenv(find_dotenv(), verbose=True) #读取本地的.env文件,里面定义了OPENAI_API_KEY等环墶变量

'''
创建一个 OpenAI 客户端实例，用于调用 OpenAI 的 API。
这里假设 OpenAI 是一个已经导入的类（可能来自 OpenAI 官方 SDK），它会自动读取环境变量中的 OPENAI_API_KEY 来进行身份验证
'''
client = OpenAI()

#文档的加载与切割
def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''从PDF文件中(按指定页码)提取文字'''
    paragraphs = [] #存储提取的段落
    buffer = '' # 用于临时存储段落内容
    full_text = '' # 用于存储所有提取的文本
    #提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)): #遍历每一页
        #如果指定了页码范围，跳过范围外的页码
        if page_numbers is not None and i not in page_numbers:
            continue

        for element in page_layout:
            if isinstance(element, LTTextContainer):#如果是文本容器
                # 提取文本  这里element就是一个文本段落，每一行的行尾都默认有换行符，full_text是一个文本段落的集合,每个文本段落之间先用换行符分隔做标记
                #将每个文本段落的文本提取出来，存储在full_text中，每个文本段落之间用换行符分隔
                full_text += element.get_text() + '\n'
    #按空行分隔，将文本重新组织成段落
    # 按换行符分割文本，得到每一行的文本字符串，lines是一个文本行的集合。
    #因为前面对于段落的提取用换行符做了分隔，所以这里split是去掉了每一行的换行符，但是保留了段落之间的换行符
    #用于下面将文本重新组织成一个个的段落
    lines = full_text.split('\n')
    for text in lines:#遍历每一行文本，text是每一行的文本
        if len(text) > min_line_length:
            # 如果文本长度大于min_line_length，将文本添加到buffer中，如果文本不是以'-'结尾，添加一个空格
            # 如果文本是以'-'结尾，去掉'-',用于将一个段落的文本拼接在一起成一个长的文本字符串
            #这里没有连接符，默认都会在每一行的行首添加一个空格，这样就会导致段落的第一行的行首会有一个空格，这样是不准确的
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            # 如果是空行或者是长度小于min_line_length的行，将buffer中的文本添加到paragraphs中，并清空buffer，
            # 这里会丢弃空行或者长度小于min_line_length的行，应该会导致不准确，会丢失一些文本行(字数少的文本行会被直接丢弃)
            paragraphs.append(buffer)#将buffer中的文本添加到paragraphs中,buffer是一个拼接好的段落
            buffer = ''
    if buffer:#用于文章的最后一个段落的最后一行符合要求的文本
        paragraphs.append(buffer)
    return paragraphs

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


if __name__ == '__main__':
    paragraphs = extract_text_from_pdf(
        "llama2.pdf",
        page_numbers=[2, 3],
        min_line_length=10
    )

    #创建一个向量数据库对象
    vector_db = MyVectorDBConnector('demo', get_embeddings)
    #向向量数据库中添加文档
    vector_db.add_document(paragraphs)

    user_query = "Llama 2有多少参数"
    #user_query = "Does Llama 2 have a conversation variant"
    results = vector_db.search(user_query, 2)

    for para in results['documents'][0]:
        print(para + "\n")