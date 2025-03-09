import json
import os
import sys
import time
import streamlit as st
import tiktoken
from openai import OpenAI

# 聊天页面
def chat_page():
    st.title("Chat（文本对话）")
    # 初始化参数
    if "base_url" in st.session_state:
        base_url = st.session_state.base_url
    else:
        base_url = "https://api.openai.com/v1"

    api_key = (
        st.session_state.api_key
        if "api_key" in st.session_state and st.session_state.api_key != ""
        else None
    )
    if api_key is None:
        st.error("Please enter your API key in the home.")
        st.stop()


    #获取当前脚本文件所在的目录路径
    src_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    #读取默认配置文件
    with open(os.path.join(src_path, 'config/default.json'), 'r', encoding='utf-8') as f:
        config_defalut = json.load(f)

    # 显示配置项
    st.session_state['model_list'] = config_defalut["completions"]["models"]
    model_name = st.selectbox('Models', st.session_state.model_list, key='chat_model_name')

    # 系统提示词选项
    option = st.radio("system_prompt", ("Manual input", "prompts"), horizontal=True, index=0)
    if option == "Manual input":
        system_prompt = st.text_input('System Prompt (Please click the button "clear history" after modification.)',
                                      config_defalut["completions"]["system_prompt"])
    else:
        # 加载预设提示词
        with open(os.path.join(src_path, 'config/prompt.json'), 'r', encoding='utf-8') as f:
            masks = json.load(f)
        masks_zh = [item['name'] for item in masks['zh']]
        masks_zh_name = st.selectbox('prompts', masks_zh)
        for item in masks['zh']:
            if item['name'] == masks_zh_name:
                system_prompt = item['context']
                break
    # 是否使用默认参数
    if not st.checkbox('default param', True):
        max_tokens = st.number_input('Max Tokens', 1, 200000, config_defalut["completions"]["max_tokens"],
                                     key='max_tokens')
        temperature = st.slider('Temperature', 0.0, 1.0, config_defalut["completions"]["temperature"],
                                key='temperature')
        top_p = st.slider('Top P', 0.0, 1.0, config_defalut["completions"]["top_p"], key='top_p')
        stream = st.checkbox('Stream', config_defalut["completions"]["stream"], key='stream')
    else:
        max_tokens = config_defalut["completions"]["max_tokens"]
        temperature = config_defalut["completions"]["temperature"]
        top_p = config_defalut["completions"]["top_p"]
        stream = config_defalut["completions"]["stream"]

    # 初始化聊天记录
    if 'chat_messages' not in st.session_state:
        st.session_state['chat_messages'] = [{"role": "system", "content": system_prompt}]

    # 清除历史记录
    if st.button("clear history"):
        st.session_state.chat_messages = [{"role": "system", "content": system_prompt}]

    # 显示聊天记录
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # 处理用户输入
    if prompt := st.chat_input():
        # 如果用户在输入框中输入了内容
        try:
            # 尝试获取 OpenAI 客户端
            client = get_openai_client(base_url, api_key)
        except Exception as e:
            # 如果获取客户端失败，显示错误信息并停止程序
            st.error(e)
            st.stop()
        # 显示用户的输入内容
        st.chat_message("user").write(prompt)
        with st.chat_message('assistant'):
            # 显示一个“Thinking...”的加载动画
            with st.spinner('Thinking...'):
                # 记录开始时间
                start_time = time.time()
                try:
                    # 临时保存当前的聊天消息
                    temp_chat_messages = st.session_state.chat_messages
                    # 将用户的输入内容添加到聊天消息中
                    temp_chat_messages.append({"role": "user", "content": prompt})
                    # 调用 OpenAI 接口生成回复
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=temp_chat_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=stream
                    )
                except Exception as e:
                    # 如果调用接口失败，显示错误信息并停止程序
                    st.error(e)
                    st.stop()
                if response:
                    # 如果设置了流式传输
                    if stream:
                        # 创建一个占位符用于显示流式传输的内容
                        placeholder = st.empty()
                        streaming_text = ''
                        for chunk in response:
                            # 如果流式传输结束，跳出循环
                            if chunk.choices[0].finish_reason == 'stop':
                                break
                            # 获取当前块的内容
                            chunk_text = chunk.choices[0].delta.content
                            if chunk_text:
                                # 累加当前块的内容并更新显示
                                streaming_text += chunk_text
                                placeholder.markdown(streaming_text)
                        # 将流式传输的内容保存为最终消息
                        model_msg = streaming_text
                    else:
                        # 如果没有设置流式传输，直接获取回复内容
                        model_msg = response.choices[0].message.content
                        # 显示回复内容
                        st.markdown(model_msg)
                    # 记录结束时间
                    end_time = time.time()
                    # 将助手的回复添加到聊天消息中
                    temp_chat_messages.append({"role": "assistant", "content": model_msg})
                    # 更新会话状态中的聊天消息
                    st.session_state.chat_messages = temp_chat_messages

                    # 计算当前对话的消耗的token数
                    if config_defalut["completions"]["num_tokens"]:
                        try:
                            # 调用函数计算 token 数量
                            num_tokens = num_tokens_from_messages(st.session_state.chat_messages, model=model_name)
                            # 显示 token 数量信息
                            info_num_tokens = f"use tokens: {num_tokens}"
                            st.info(info_num_tokens)
                        except Exception as e:
                            print(e)
                    # 生成当前对话耗时信息
                    if config_defalut["completions"]["use_time"]:
                        st.info(f"Use time: {round(end_time - start_time, 2)}s")


def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        # 每个消息有一个基本的令牌数 tokens_per_message，默认3个token，每个 name 属性预设的固定令牌数 tokens_per_name，假设其值为 1。
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    #函数通过迭代消息列表，并根据消息的角色 (如 user、assistant、tool、system) 计算令牌数量。
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    # 每个回复都以 <|start|>assistant<|message|> 开头
    # 例如：<|start|>assistant<|message|>今天天气很好，适合出门！ <|end|>
    num_tokens += 3
    return num_tokens

# 使用缓存，当参数不变时，不会重复创建client
@st.cache_resource
def get_openai_client(url, api_key):
    #使用了缓存，当参数不变时，不会重复创建client
    client = OpenAI(base_url=url, api_key=api_key)
    return client

if __name__ == "__main__":
    chat_page()
