'''
LLMs(大语言模型)
'''
import os
from langchain_community.llms import QianfanLLMEndpoint
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), verbose=True)

# # 1.设置qianfan的API-KEY和Serect-KEY
# os.environ["QIANFAN_AK"] = QIANFAN_AK
# os.environ["QIANFAN_SK"] = QIANFAN_SK

# 2.实例化模型
llm = QianfanLLMEndpoint(model="Qianfan-Chinese-Llama-2-7B")
res = llm("请帮我讲一个鬼故事")
print(res)