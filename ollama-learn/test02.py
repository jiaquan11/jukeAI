#使用openai的代码风格调用ollama
from openai import OpenAI

#定义多轮对话方法
def run_chat_session():
    #初始化客户端
    client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
    #初始化对话历史
    chat_history = []
    #启动多轮对话
    while True:
        #获取用户输入
        user_input = input("用户: ")
        if user_input.lower() == "exit":
            print("退出对话")
            break
        #更新对话历史(添加用户输入)
        chat_history.append({"role": "user", "content": user_input})
        #调用模型回答
        try:
            chat_complition = client.chat.completions.create(
                messages=chat_history, model="llama3.2:1b")
            #获取最新的回答
            model_response = chat_complition.choices[0]
            print("AI: ", model_response.message.content)
            #更新对话历史(添加AI回答)
            chat_history.append({"role": "assistant", "content": model_response.message.content})
        except Exception as e:
            print("发生错误: ", e)
            break

if __name__ == "__main__":
    run_chat_session()
