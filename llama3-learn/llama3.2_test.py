#使用transformer加载llama3模型
from transformers import AutoModelForCausalLM,AutoTokenizer

DEVICE = 'cuda'
#加载本地模型路径为该模型配置文件所在的根目录
model_dir = "/root/onethingai-fs/aipro/llama3-learn/model/LLM-Research/Llama-3___2-1B-Instruct/"
#使用transformer加载模型
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

#调用模型
#定义提示词
prompt = "您好，你叫什么名字？你是由谁创造的？"
#将提示词封装成message
message = [{"role":"system", "content":"You are a helpful assistant system"}, {"role":"user","content":prompt}]
#使用分词器的apply_chat_template()方法将上面定义的消息列表进行转换,tokenize=False表示此时不进行令牌化
text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

#将处理后的文本令牌化冰转换为模型的输入张量
model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
#输入模型得到输出
generated = model.generate(model_inputs.input_ids, max_new_tokens=512)
print(generated)

#对输出的内容进行解码还原
response = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(response)