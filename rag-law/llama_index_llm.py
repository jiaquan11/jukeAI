from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM

#使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(model_name="/home/jukeai/ai_projects/llm/Qwen/Qwen1.5-7B-Chat",
               tokenizer_name="/home/jukeai/ai_projects/llm/Qwen/Qwen1.5-7B-Chat",
               model_kwargs={"trust_remote_code":True},
               tokenizer_kwargs={"trust_remote_code":True})
#调用模型chat引擎得到回复
rsp = llm.chat(messages=[ChatMessage(content=" 哪种情况下用人单位不得解除劳动合同？")])

print(rsp)