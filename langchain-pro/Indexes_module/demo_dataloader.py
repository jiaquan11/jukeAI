from langchain_community.document_loaders import UnstructuredFileLoader

# 实例话UnstructuredFileLoader
loader = UnstructuredFileLoader("./衣服属性.txt", encoding='utf8')
docs = loader.load()
print(f'docs-->{docs}')
print(len(docs))
print(docs[0].page_content[:10])

print("*"*80)
from langchain_community.document_loaders import TextLoader

loader1 = TextLoader("./衣服属性.txt", encoding='utf8')
doc1 = loader1.load()
print(f'docs1-->{doc1}')
print(len(doc1))
print(doc1[0].page_content[:10])