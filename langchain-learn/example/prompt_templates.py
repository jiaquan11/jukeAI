from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
result = prompt_template.invoke({"language": "Chinese", "text": "hi"})
print(result.to_messages())
# [SystemMessage(content='Translate the following into Chinese:'), HumanMessage(content='hi')]

parser = StrOutputParser()

# 使用Chains方式调用
chain = prompt_template | model | parser
response = chain.invoke({"language": "Chinese", "text": "hi"})
print(response)
# 你好
