from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="将以下内容从英语翻译成中文"),
    HumanMessage(content="It's a nice day today"),
]
parser = StrOutputParser()
result = model.invoke(messages)
print(result)
#使用parser处理model返回的结果
response = parser.invoke(result)
print(response)
# 今天天气很好
