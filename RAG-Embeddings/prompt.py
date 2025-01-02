from openai import OpenAI
import os

#加载环境变量
'''
使用 python-dotenv 库加载 .env 文件中的环境变量。
.env 文件通常用于存储敏感信息，例如 API 密钥（OPENAI_API_KEY）。
find_dotenv()：
自动查找项目中的 .env 文件路径。
load_dotenv()：
加载 .env 文件中的环境变量到系统环境中。
verbose=True：
如果设置为 True，会输出加载 .env 文件的详细信息，方便调试。
'''
from dotenv import load_dotenv, find_dotenv
from pyexpat.errors import messages

_ = load_dotenv(find_dotenv(), verbose=True) #读取本地的.env文件,里面定义了OPENAI_API_KEY等环墶变量

'''
创建一个 OpenAI 客户端实例，用于调用 OpenAI 的 API。
这里假设 OpenAI 是一个已经导入的类（可能来自 OpenAI 官方 SDK），它会自动读取环境变量中的 OPENAI_API_KEY 来进行身份验证
'''
client = OpenAI()

def get_completion(prompt, model="gpt-4o"):
    '''封装openai接口'''
    messages = [{"role":"user", "content":prompt}]
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        #模型输出的随机性，0表示随机性最小，1表示随机性最大
        temperature=0.7,#温度参数，用于控制生成文本的多样性，值越大，生成的文本越多样
    )
    return response.choices[0].message.content

'''
作用：
根据 prompt_template 和传入的参数（kwargs），动态生成一个完整的 prompt。
参数：
prompt_template：一个字符串模板，包含占位符（如 {context} 和 {query}）。
kwargs：传入的键值对，用于替换模板中的占位符。
实现细节：
遍历 kwargs 中的键值对：
如果值是字符串列表（list[str]），将列表中的字符串用 \n\n 拼接成一个字符串。
否则，直接使用值。
使用 prompt_template.format(**inputs) 将占位符替换为实际值，生成最终的 prompt。
'''
def build_prompt(prompt_template, **kwargs):
    '''将Prompt模版赋值'''
    inputs = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val
    return prompt_template.format(**inputs)

prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。

已知信息:
{context} # 检索出来的原始文档

用户问：
{query} # 用户的提问

如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
请不要输出已知信息中不包含的信息或答案。
请用中文回答用户问题。
"""

if __name__ == '__main__':
    #直接调用 get_completion 测试简单的问答功能
    #prompt = "请用中文解释什么是人工智能。"
    #response = get_completion(prompt)
    #print(response)  # 输出生成的回答

    #使用 build_prompt 构建复杂的 prompt，测试基于上下文的问答功能
    context = "人工智能是一种模拟人类智能的技术，广泛应用于自然语言处理、计算机视觉等领域。"
    query = "人工智能有哪些应用？"
    prompt = build_prompt(prompt_template, context=context, query=query)
    response = get_completion(prompt)
    print(response)