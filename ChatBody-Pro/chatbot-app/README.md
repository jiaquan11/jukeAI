# chatbot-app

基于streamlit编写的针对openai接口的各类模型对话web应用，目前支持基本对话、文生图、图片理解、语音转文、文转语言。


## 0. 目标
- [x] 支持Chat Completions API 的页面，进行文本对话
- [x] 支持Vision的页面，进行图像理解
- [x] 支持Image generation的页面，进行图片生成
- [x] 支持语言转文本的页面
- [x] 支持文本转语音的页面 

其他添加的小功能
- [x] chat页面每次对话后显示耗时（配置文件中控制开关）
- [x] chat页面每次对话后显示消耗的token数（配置文件中控制开关）
- [x] 对whisper的输入添加违禁词过滤，包含违禁词则输出null
- [x] 给chat页面增加了预设提示词，可以在配置文件中自定义
- [x] 在draw页面添加了`revised_prompt`的显示，它保存了对你提示词的修改结果


## 1 使用

### 直接运行

```bash
pip install -r requirements.txt
pip install --upgrade streamlit audio_recorder_streamlit
#命令行启动
streamlit run ./src/home.py --server.port 1234
#PyCharm debug启动
script：D:/software/python/python-3.12.4/.venv/Scripts/streamlit
script parameters：run ./src/home.py --server.port 1234
Working directory：./chatbot-app
```

## 2. 配置文件
默认参数配置文件在`src/config/default.json`中，这里主要说明自定义的参数：
* models：chat页面上的下拉菜单显示的模型
* num_tokens：是否显示每次对话消耗的token数
* use_time：是否显示每次对话消耗的时间