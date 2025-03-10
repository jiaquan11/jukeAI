from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])
result = prompt_template.invoke({"msgs": [HumanMessage(content="hi!"),HumanMessage(content="hello!")]})
#print(result)
# messages=[SystemMessage(content='You are a helpful assistant'), HumanMessage(content='hi!')]

# 另一种实现相同效果的替代方法是，不直接使用 MessagesPlaceholder 类，而使用placeholder
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("placeholder", "{msgs}")  # <-- 这是更改的部分
])
# messages=[SystemMessage(content='You are a helpful assistant'), HumanMessage(content='hi!')]

result = prompt_template.invoke({"msgs": [HumanMessage(content="hi!"),HumanMessage(content="hello!")]})
print(result)