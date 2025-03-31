from llama_index.llms.openai_like import OpenAILikeLLM
from llama_index.core import Settings

# 配置vLLM服务端参数
class VLLMConfig:
    API_BASE = "http://localhost:8000/v1"  # vLLM的默认端点
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1___5B"
    API_KEY = "no-key-required"  # vLLM默认不需要密钥
    TIMEOUT = 60  # 请求超时时间

# 初始化LLM（替换原来的HuggingFaceLLM）
def init_vllm_llm():
    return OpenAILikeLLM(
        model=VLLMConfig.MODEL_NAME,
        api_base=VLLMConfig.API_BASE,
        api_key=VLLMConfig.API_KEY,
        temperature=0.3,
        max_tokens=1024,
        timeout=VLLMConfig.TIMEOUT,
        is_chat_model=True,  # 适用于对话模型
        additional_kwargs={"stop": ["<|im_end|>"]}  # DeepSeek的特殊停止符
    )

# 在全局设置中配置
Settings.llm = init_vllm_llm()