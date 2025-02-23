import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), verbose=True)

# 创建第一条链
#1. 定义模板
template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
prompt = PromptTemplate(input_variables=["lastname"], template=template)

# 2.实例化模型
llm = QianfanLLMEndpoint()

# 3.构造Chain：第一条链
first_chain = LLMChain(llm=llm, prompt=prompt)

# 创建第二条链
#1. 定义模板
second_prompt = PromptTemplate(input_variables=["child_name"],
                               template="邻居的儿子名字叫{child_name}，给他起一个小名")

#2. 创建第二条链
second_chain = LLMChain(llm=llm, prompt=second_prompt)

# 融合两条链:verbose为True的时候，显示模型推理过程，否则不显示
overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)

# 使用链
result = overall_chain.run("王")
print(result)