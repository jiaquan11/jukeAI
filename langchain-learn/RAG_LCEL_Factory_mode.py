#用LCEL实现工厂模式
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.utils import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 模型1
ernie_model = QianfanChatEndpoint(
    qianfan_ak=os.getenv('ERNIE_CLIENT_ID'),
    qianfan_sk=os.getenv('ERNIE_CLIENT_SECRET')
)

print("ERNIE_CLIENT_ID:", os.getenv('ERNIE_CLIENT_ID'))
print("ERNIE_CLIENT_SECRET:", os.getenv('ERNIE_CLIENT_SECRET'))

# 模型2
gpt_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 通过 configurable_alternatives 按指定字段选择模型
model = gpt_model.configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="gpt",
    ernie=ernie_model,
    # claude=claude_model,
)

# Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

# LCEL
chain = (
    {"query": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 运行时指定模型 "gpt" or "ernie"
ret = chain.with_config(configurable={"llm": "ernie"}).invoke("请自我介绍")
print(ret)