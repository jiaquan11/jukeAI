from langchain_community.document_loaders import PyMuPDFLoader

# 1、通过PyMuPDFLoader加载PDF文档
loader = PyMuPDFLoader("llama2.pdf")
pages = loader.load_and_split()

print(pages[0].page_content) #输出第一页的内容

# 2、通过RecursiveCharacterTextSplitter切割文本
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 简单的文本内容切割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
paragraphs = text_splitter.create_documents([pages[0].page_content])
for para in paragraphs:
    print(para.page_content)
    print('-------')