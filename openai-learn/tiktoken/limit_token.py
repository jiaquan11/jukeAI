from openai import OpenAI
# pip install tiktoken
import tiktoken

client = OpenAI()

# 这是 API 请求和响应的总 token 数量限制。对于 GPT-4 模型，这个值通常是 4096。
MAX_TOKENS = 8  # 设置最大 token 数量
# 这是我们预留给模型响应的 token 数量。我们需要在计算对话的最大 token 数量时减去这个值，以确保有足够的空间来容纳模型的响应。
MAX_RESPONSE_TOKENS = 6  # 设置响应中预留的最大 token 数量
encoder = tiktoken.encoding_for_model("gpt-4")
def count_tokens(text):
    encoder.encode(text)
    # 将输入的文本text转换为对应的token列表。具体来说，它使用tiktoken库中的编码器将文本进行编码，以便后续处理。
    tokens = encoder.encode(text)
    # 统计文本中的 token 数量
    return len(tokens)
# 假设 MAX_TOKENS 是 4096，而 MAX_RESPONSE_TOKENS 是 500，那么：
# 我们希望对话历史的 token 数量不要超过 3596 (4096 - 500)。
# 这样，当我们发送对话历史给 API 时，仍然有 500 个 token 的空间用于模型生成的响应。
def manage_token_limit(messages):
    current_tokens = count_tokens(messages)
    if current_tokens > (MAX_TOKENS - MAX_RESPONSE_TOKENS):
        print(f"当前会话 token 数量: {current_tokens}, 超过最大 token 数量: {MAX_TOKENS - MAX_RESPONSE_TOKENS}")
        return False
    return True


def get_gpt_response(messages):
    """获取 GPT-4 的响应"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message.content.strip()


def main():
    messages = []

    print("Chat with GPT-4. Type 'exit' to end the conversation.")
    while True:
        user_input = input("用户: ")
        if user_input.lower() == 'exit':
            break

        messages.append({"role": "user", "content": user_input})

        # 管理用户输入以确保总 token 数量不超过限制
        if not manage_token_limit(user_input):
            continue

        response = get_gpt_response(messages)
        print(f"GPT: {response}")

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
