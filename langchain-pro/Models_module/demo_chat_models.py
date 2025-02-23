'''
Chat Models(聊天模型)
'''
import os
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), verbose=True)


# 2.实例化模型
chat = QianfanChatEndpoint(model='Qianfan-Chinese-Llama-2-7B')
messages = [HumanMessage(content="给我写一首唐诗")]
res = chat(messages)
print(res)