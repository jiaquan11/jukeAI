import os

import streamlit as st

def home():
    st.title("🏠openai playground")
    st.caption("Please fill in the parameters in the sidebar before using, or import the parameters by uploading a file.")

    if "base_url" not in st.session_state:
        st.session_state['base_url'] = os.getenv('OPENAI_BASE_URL')
    
    if "api_key" not in st.session_state:
        st.session_state['api_key'] =  os.getenv('OPENAI_API_KEY')


    ## 输入方式
    st.session_state.base_url = st.sidebar.text_input('Base URL', st.session_state.base_url)
    st.session_state.api_key =  st.sidebar.text_input('API Key',st.session_state.api_key, type='password')

    option = st.radio("change language:", ("En", "Zh"),horizontal=True,index=1)
    if option == "Zh":
        st.markdown(
                """
                **体验OpenAI多模态功能**
                ## 使用说明
                * 请在侧边栏填写`API Key`，如果没有请在[OpenAI官网](https://platform.openai.com/account/api-keys)获取，如果需要使用代理，请修改`base_url`\n
                * 接下来在侧边栏选择需要使用的页面。
                ---------------------------------------------------------
                """
        )
        st.markdown(
                """
                ### 1 💬chat （文本对话）  \n
                该页面用于文本对话，选择模型，输入问题，得到回答。对应openai文档：[text-generation](https://platform.openai.com/docs/guides/text-generation)\n
                
                ### 2 🖼️drawing（文生图） \n
                该页面用于图像生成，使用DALL·E模型，输入提示词，输出图片。对应openai文档：[image-generation](https://platform.openai.com/docs/guides/images?context=node)\n

                ### 3 🗣️speech to text（语音转文本）\n
                该页面用于语音转文本，使用whisper模型。对应openai文档：[speech-to-text](https://platform.openai.com/docs/guides/speech-to-text)\n
                
                ### 4 📢text to speech（文本转语音）\n
                该页面用于文本转语音，使用tts模型。对应openai文档：[text-to-speech](https://platform.openai.com/docs/guides/text-to-speech)\n
                
                ### 5 🎞️vision（图像理解） \n
                该页面用于图像理解，使用gpt-4o模型，输入图片和问题，得到回答。对应openai文档：[vision](https://platform.openai.com/docs/guides/vision)\n
                
                """
            )
    elif option == "En":
            st.markdown(
            """
            **Here you can experience all the capabilities provided by OpenAI.**
            ## Instructions for use
            * Please fill in the `API Key` in the sidebar. If you don't have one, you can obtain it from the [OpenAI website](https://platform.openai.com/account/api-keys). If you need to use a proxy, please modify the `base_url`.
            * You can also automatically populate the fields by importing a JSON file with the following format:
            ```json
            {
                "base_url" : "https://xxx",
                "api_key" : "sk-xxxx" 
            }
            ```
            * Next, select the desired page from the sidebar.
            ---------------------------------------------------------
            ### 1 💬chat page
            This page is used for text-based conversations. Select a model, input a question, and get a response. Corresponds to the OpenAI documentation: [text-generation](https://platform.openai.com/docs/guides/text-generation)

            ### 2 🎞️vision page
            This page is used for image understanding. It utilizes the gpt-4-vision-preview model. Input an image and a question, and get a response. Corresponds to the OpenAI documentation: [vision](https://platform.openai.com/docs/guides/vision)

            ### 3 🖼️drawing page
            This page is used for image generation. It utilizes the DALL·E model. Input prompts and generate images. Corresponds to the OpenAI documentation: [image-generation](https://platform.openai.com/docs/guides/images?context=node)

            ### 4 🗣️speech to text
            This page is used for speech-to-text conversion. It utilizes the whisper model. Corresponds to the OpenAI documentation: [speech-to-text](https://platform.openai.com/docs/guides/speech-to-text)

            ### 5 📢text to speech
            This page is used for text-to-speech conversion. It utilizes the tts model. Corresponds to the OpenAI documentation: [text-to-speech](https://platform.openai.com/docs/guides/text-to-speech)
            """
        )
if __name__ == "__main__":
    home()