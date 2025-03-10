from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])
result = prompt_template.invoke({"topic": "cats"})
print(result)
#messages=[SystemMessage(content='You are a helpful assistant'), HumanMessage(content='Tell me a joke about cats')]