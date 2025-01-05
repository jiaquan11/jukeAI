from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader

#langchain 向量数据库与向量检索
# 加载文档
loader = PyMuPDFLoader("llama2.pdf")
pages = loader.load_and_split()

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents(
    [page.page_content for page in pages[:4]] # 只取前4页
)

# 灌库
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = FAISS.from_documents(texts, embeddings) # 使用FAISS作为向量检索引擎

# 检索 top-3 结果
retriever = db.as_retriever(search_kwargs={"k": 3}) # 检索器

docs = retriever.invoke("llama2有多少参数")
for doc in docs:
    print(doc.page_content)
    print("----")