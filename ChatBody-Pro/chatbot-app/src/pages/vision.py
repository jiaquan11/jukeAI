from openai import OpenAI
import streamlit as st
import json
import os, sys  # os 和 sys 用于与操作系统和系统参数交互。
import base64  # base64 用于处理 Base64 编码
import requests  # requests 用于发送 HTTP 请求

@st.cache_resource
def get_openai_client(url, api_key):
    # 使用了缓存，当参数不变时，不会重复创建client
    client = OpenAI(base_url=url, api_key=api_key)
    return client

# 这段代码定义了一个名为 vision_page 的函数，并设置了页面标题和描述，解释了 GPT-4o 的功能及其当前的限制。
def vision_page():
    st.title("GPT-4o（图像理解）")
    st.caption("它接受文本、音频和图像的任何组合作为输入，并生成文本、音频和图像的任何组合作为输出")
    st.caption("目前gpt-4o 的api接口还不支持音频输入，只能传入图片，所以就和gpt-4-vision-preview差不多了")
    # 初始化参数
    api_key = (
        st.session_state.api_key
        if "api_key" in st.session_state and st.session_state.api_key != ""
        else None
    )
    if api_key is None:
        st.error("Please enter your API key in the home.")
        st.stop()

    if "base_url" in st.session_state:
        base_url = st.session_state.base_url
    else:
        base_url = "https://api.openai.com/v1"

    src_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    with open(os.path.join(src_path, 'config/default.json'), 'r', encoding='utf-8') as f:
        config_defalut = json.load(f)

    # 读取配置文件 default.json 并将其中的模型列表存储在 session_state 中。
    st.session_state['model_list'] = config_defalut["completions"]["models"]

    # 创建一个文件上传器，允许用户上传图片文件，并设置最大文件大小为 5MB。
    upload_images = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    # 创建一个数字输入框，让用户输入最大 tokens 数量，默认值为 300。
    max_tokens = st.number_input("Max tokens", min_value=1, value=300, step=1)

    # 检查上传的图片文件是否超过最大大小，如果没有超过，则读取文件内容并显示图片。
    bytes_data = None
    if upload_images is not None:
        if upload_images.size > MAX_FILE_SIZE:
            st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            # image = Image.open(upload_images)
            bytes_data = upload_images.getvalue()
            st.image(bytes_data, caption=upload_images.name, width=200)

    # 处理用户输入的提示信息 prompt 和上传的图片
    if prompt := st.chat_input():
        #如果用户输入了提示信息，则显示用户消息。
        st.chat_message("user").write(prompt)
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                try:
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                    #如果上传了图片，则将图片转换为 Base64 编码，并构建包含文本和图片的请求负载。
                    if bytes_data is not None:
                        base64_image = base64.b64encode(bytes_data).decode("utf-8")
                        payload = {
                            "model": "gpt-4o",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": prompt,
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                        },
                                    ],
                                },
                            ],
                            "max_tokens": max_tokens,
                        }
                    else:
                        #如果没有上传图片，则构建仅包含文本的请求负载。
                        payload = {
                            "model": "gpt-4o",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt,
                                },
                            ],
                            "max_tokens": max_tokens,
                        }
                    #发送请求到 OpenAI API
                    if base_url.endswith('/'):
                        base_url = base_url[:-1]
                    response = requests.post(
                        base_url + "/chat/completions", headers=headers, json=payload
                    )
                    # 检查状态码是否正常，不正常会触发异常
                    response.raise_for_status()
                    print(response.json())
                    result = response.json()["choices"][0]["message"]["content"]
                    st.markdown(result)
                except Exception as e:
                    st.error(e)
                    st.stop()

if __name__ == "__main__":
    vision_page()
