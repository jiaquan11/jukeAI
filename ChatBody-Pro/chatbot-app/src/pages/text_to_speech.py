import streamlit as st
from openai import OpenAI

# 定义一个函数，用于创建 TTS 页面。
def tts_page():
    # 设置页面标题。
    st.title("text to speech（文本转语音）")
    # 添加页面说明文字
    st.caption(
        "基于TTS（文本到语音）模型，有6个内置声音")
    # 初始化参数，尝试从 st.session_state 中获取 API 密钥，如果没有则设置为 None。
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

    client = get_openai_client(base_url, api_key)

    # 创建一个下拉菜单供用户选择模型。
    model = st.selectbox('model ', ["tts-1", "tts-1-hd"])
    # 创建一个下拉菜单供用户选择发音人
    voice = st.selectbox('voice ', ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
    # 建一个滑动条供用户选择语速。
    speed = st.slider('speed', 0.25, 4.0, 1.0)

    option = st.radio("Input methods:", ("Manual input", "import document"), horizontal=True, index=0)
    # 如果用户选择导入文档方式
    if option == "import document":
        # 创建一个文件上传控件，允许用户上传 .txt 文件
        uploaded_file = st.file_uploader("Choose a file", type=["txt"])
        # 初始化 content 变量
        content = None
        # 如果用户上传了文件
        if uploaded_file is not None:
            # 读取文件内容并解码为 UTF-8
            content = uploaded_file.read().decode("utf-8")
            # 显示文件内容
            st.text(content)
        # 如果用户点击了“generated”按钮并且文件内容不为空：
        if st.button("generated") and content:
            # 显示加载动画
            with st.spinner('doing...'):
                # 调用 generated_speech 函数生成语音文件。
                speech_file_path = generated_speech(client, model, voice, content, speed)
                # 播放生成的语音文件
                st.audio(speech_file_path)

    # 如果用户选择手动输入方式
    else:
        # 如果用户在聊天输入框中输入了内容：
        if prompt := st.chat_input("prompt"):
            # 显示用户输入的内容
            st.chat_message("user").write(prompt)
            # 显示助手的消息
            with st.chat_message('assistant'):
                # 显示加载动画
                with st.spinner('doing...'):
                    try:
                        # 调用 generated_speech 函数生成语音文件
                        speech_file_path = generated_speech(client, model, voice, prompt, speed)
                        # 播放生成的语音文件。
                        st.audio(speech_file_path)
                    except Exception as e:
                        st.error(e)
                        st.stop()


# 定义一个函数，用于生成语音文件
def generated_speech(client, model, voice, prompt, speed):
    # 导入 tempfile 模块，用于创建临时文件。
    import tempfile
    # 创建一个临时文件，后缀为 .mp3，并且在文件关闭后不删除。
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        # 获取临时文件的路径
        speech_file_path = temp_file.name
        # 使用 OpenAI 客户端生成语音，传入模型、声音、输入文本和速度参数。
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=prompt,
        speed=speed,
    )
    # 将生成的语音流写入临时文件。
    response.stream_to_file(speech_file_path)
    # 返回生成的语音文件路径
    return speech_file_path


@st.cache_resource
def get_openai_client(url, api_key):
    client = OpenAI(base_url=url, api_key=api_key)
    return client


if __name__ == "__main__":
    tts_page()
