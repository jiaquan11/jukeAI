from langchain_openai import ChatOpenAI
import asyncio
model = ChatOpenAI(model="gpt-4")

# 异步流处理
async def async_stream():
    events = []
    async for event in model.astream_events("hello", version="v2"):
        events.append(event)
    print(events)

# 运行异步流处理
asyncio.run(async_stream())
