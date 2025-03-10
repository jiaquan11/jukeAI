#pip install langchain-core
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
result = prompt_template.invoke({"topic": "cats"})
print(result)