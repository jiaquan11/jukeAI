from http.client import responses
from itertools import product

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

#format输出
'''
#OpenAI模型封装
llm = ChatOpenAI(model="gpt-4o-mini") #默认是gpt-3.5-turbo模型
response = llm.invoke("你是谁")
print(response.content)
'''

'''
#多轮对话Session封装
from langchain.schema import (
    AIMessage,  # 等价于OpenAI接口中的assistant role
    HumanMessage,  # 等价于OpenAI接口中的user role
    SystemMessage  # 等价于OpenAI接口中的system role
)

messages = [
    SystemMessage(content="你是聚客AI研究院的课程助理。"),
    HumanMessage(content="我是学员，我叫大拿。"),
    AIMessage(content="欢迎！"),
    HumanMessage(content="我是谁")
]

ret = llm.invoke(messages)
print(ret.content)
'''

'''
#Prompt模板封装
#PromptTemplate可以在模版中自定义变量
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template("给我讲个关于{subject}的笑话") #定义模版
print("===Template===")
print(template)
print("===Prompt===")
print(template.format(subject="小明"))

from langchain_openai import ChatOpenAI
#定义LLM
llm = ChatOpenAI(model="gpt-4o-mini")
#通过Prompt调用LLM
ret = llm.invoke(template.format(subject="小明"))
#打印输出
print(ret.content)
'''

'''
#ChatPromptTemplate用模版表示的对话上下文
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,)
from langchain_openai import ChatOpenAI

template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是{product}的客服助手，你的名字叫{name}"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = template.format_messages(product="聚客AI研究院", name="大吉", query="你是谁")
print (prompt)
ret = llm.invoke(prompt)
print(ret.content)
'''

'''
#MessagesPlaceholder把多轮对话变成模版
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

human_prompt = "Translate your answer to {language}."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    # variable_name 是 message placeholder 在模板中的变量名
    # 用于在赋值时使用
    [MessagesPlaceholder("history"), human_message_template]
)

from langchain_core.messages import AIMessage, HumanMessage

human_message = HumanMessage(content="Who is Elon Musk?")
ai_message = AIMessage(
    content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer"
)

messages = chat_prompt.format_prompt(
    # 对 "history" 和 "language" 赋值
    history=[human_message, ai_message], language="中文"
)

print(messages.to_messages())

llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke(messages)
print(result.content)
'''

'''
#把prompt模版看作带有参数的函数
#从文件加载prompt模版
from langchain.prompts import PromptTemplate

template = PromptTemplate.from_file("example_prompt_template.txt", encoding="utf-8")
print("===Template===")
print(template)
print("===Prompt===")
print(template.format(topic='黑色幽默'))
'''

'''
#结构化输出
#直接输出Pydantic对象
from pydantic import BaseModel, Field
# 定义你的输出对象
class Date(BaseModel):
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="BC or AD")

from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

model_name = 'gpt-4o-mini'
temperature = 0
llm = ChatOpenAI(model_name=model_name, temperature=temperature)

# 定义结构化输出的模型
structured_llm = llm.with_structured_output(Date)

template = """提取用户输入中的日期。
用户输入:
{query}"""

prompt = PromptTemplate(
    template=template,
)

query = "2024年十二月23日天气晴..."
input_prompt = prompt.format_prompt(query=query)
result = structured_llm.invoke(input_prompt)
print("result content1:")
print(result)
print("result content2:")

#输出指定格式的JSON
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

json_schema = {
    "title": "Date",
    "description": "Formated date expression",
    "type": "object",
    "properties": {
        "year": {
            "type": "integer",
            "description": "year, YYYY",
        },
        "month": {
            "type": "integer",
            "description": "month, MM",
        },
        "day": {
            "type": "integer",
            "description": "day, DD",
        },
        "era": {
            "type": "string",
            "description": "BC or AD",
        },
    },
}

# 定义结构化输出的模型
structured_llm = llm.with_structured_output(json_schema) #指定输出格式
print("result content 3:")
result = structured_llm.invoke(input_prompt)
print(result)
print("result content 4:")

#OutputParser解析,可以按指定格式解析模型的输出
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser(pydantic_object=Date)
prompt = PromptTemplate(
    template="提取用户输入中的日期。\n用户输入:{query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

input_prompt = prompt.format_prompt(query=query)
output = llm.invoke(input_prompt)
print("原始输出:\n"+output.content)
print("\n解析后:")
result = parser.invoke(output)
print(result)

#也可以用PydanticOutputParser解析
from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=Date)
input_prompt = prompt.format_prompt(query=query)
output = llm.invoke(input_prompt)
print("原始输出:\n"+output.content)
print("\n解析后:")
result = parser.invoke(output)
print(result)

#OutputFixingParser解析,利用大模型做格式自动纠错
from langchain.output_parsers import OutputFixingParser
new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(model="gpt-4o"))
bad_output = output.content.replace("4","四")
print("PydanticOutputParser:")
try:
    result = parser.invoke(bad_output) #不会自动纠错,会报错
    print(result)
except Exception as e:
    print(e)

print("OutputFixingParser:")
result = new_parser.invoke(bad_output) #自动纠错
print(result)
'''

#Function Calling
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

import json

model_name = 'gpt-4o-mini'
temperature = 0
llm = ChatOpenAI(model_name=model_name, temperature=temperature)

llm_with_tools = llm.bind_tools([add, multiply])

query = "3的4倍是多少?"
messages = [HumanMessage(query)]

output = llm_with_tools.invoke(messages)

print(json.dumps(output.tool_calls, indent=4))

#回传Function Call的结果
messages.append(output)

available_tools = {"add": add, "multiply": multiply}

for tool_call in output.tool_calls:
    selected_tool = available_tools[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

new_output = llm_with_tools.invoke(messages)
for message in messages:
    print(json.dumps(message.model_dump(), indent=4, ensure_ascii=False))
print(new_output.content)