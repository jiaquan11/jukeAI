import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), verbose=True)

# 实例化模型
llm = QianfanLLMEndpoint(model="Qianfan-BLOOMZ-7B-compressed")

# 定义工具:这里指定两个工具来选择使用：llm-math计算，wikipedia
tools = load_tools(["wikipedia"], llm=llm)

# 实例化agent
agent = initialize_agent(tools=tools,
                         llm=llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)

#print(f'agent-->{agent}')

prompt_template = "中国目前有多少人口"
prompt = PromptTemplate.from_template(prompt_template)

# 执行代理
result = agent.run(prompt)
print(result)