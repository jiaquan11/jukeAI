from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-4")

# 异步流处理
async def async_stream():
    chunks = []
    async for chunk in model.astream("天空是什么颜色？"):
        chunks.append(chunk)
        # 判断chunks长度为1的时候，打印chunks[0]
        if len(chunks) == 2:
            print(chunks[1])
        print(chunk.content, end="|", flush=True)
# 运行异步流处理
asyncio.run(async_stream())
