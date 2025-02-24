from PyPDF2 import PdfReader # 读取pdf文件
from langchain.text_splitter import CharacterTextSplitter # 文本分割器
from langchain_community.vectorstores import FAISS # 向量库
from langchain_community.llms import QianfanLLMEndpoint # 千帆大模型平台库
from langchain.llms import OpenAI # openai模型库
import streamlit as st # 搭建web界面
from langchain.chains import ConversationalRetrievalChain # 对话检索链
import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings #向量模型
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), verbose=True)

# 使用国外OPENAI的模型，需要导入API-KEY(需要科学上网)
# os.environ[ "OPENAI_API_KEY" ] = OPENAI_AK
# 使用国内百度千帆平台的模型，需要导入API-KEY和SERECT-KEY
# 设置web页面：比如标题、描述功能
st.title("《英雄玩法》")
st.write("请上传一个关于英雄攻略介绍的pdf文档.")
# 设置上传pdf文件的功能
uploaded_file = st.file_uploader("选择一个pdf文档", type="pdf")

if uploaded_file:
# 读取pdf文件
    doc_reader = PdfReader(uploaded_file)
# 从pdf中提取文档
    raw_text = ""
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    #将文本切分成小的模块
    # print(raw_text)
    # print('*'*80)
    text_splitter = CharacterTextSplitter(separator="。",chunk_size=100, chunk_overlap=10)

    texts = text_splitter.split_text(raw_text)
    # print(texts)
    # print(f'len(texts)-->{len(texts[0])}')
    # print('*'*80)
    # embeddings模型
    # EMBEDDING_MODEL = "/Users/ligang/PycharmProjects/llm/langchain_apply/Knowledge_QA/moka-ai/m3e-base"
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    embeddings = QianfanEmbeddingsEndpoint()
    # 创建文档搜索
    docsearch = FAISS.from_texts(texts, embeddings)
    # 创建对话链
    qa = ConversationalRetrievalChain.from_llm(
                        llm=QianfanLLMEndpoint(model='Qianfan-Chinese-Llama-2-7B'), # llm=OpenAI(model='gpt-3.5-turbo'),
                        retriever=docsearch.as_retriever(),
                        return_source_documents=True, )
                        # 初始化聊天记录列表
    chat_history = []
                        # 获取用户的查询
    query = st.text_input("请给出你的问题")
    #添加一个生成按钮
    generate_button = st.button("生成答案")
    if generate_button and query:
        with st.spinner("答案生成中..."):
        # 将问题以及历史对话记录传入对话链获得模型输出结果
            result = qa({"question": query, "chat_history": chat_history})
            answer = result["answer"]
            source_documents = result['source_documents']
            #将答案和source_documents合并为单个响应（输出）
            response = { "answer": answer, "source_documents": source_documents
            }
            st.write("response:", response)