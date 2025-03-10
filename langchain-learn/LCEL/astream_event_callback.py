from langchain_openai import ChatOpenAI
import asyncio
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

model = ChatOpenAI(model="gpt-4")

# 异步流处理
async def async_stream():
    def reverse_word(word: str):
        return word[::-1]

    reverse_word = RunnableLambda(reverse_word)

    @tool
    def bad_tool(word: str):
        """不传播回调的自定义工具。"""
        return reverse_word.invoke(word)

    async for event in bad_tool.astream_events("hello", version="v2"):
        print(event)

# 运行异步流处理
asyncio.run(async_stream())
