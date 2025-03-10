from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="将以下内容从英语翻译成中文"),
    HumanMessage(content="Let's go for a run"),
]
parser = StrOutputParser()

# 使用Chains方式调用
chain = model | parser
response = chain.invoke(messages)
print(response)
#我们去跑步吧
