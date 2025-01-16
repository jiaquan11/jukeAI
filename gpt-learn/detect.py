from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

#加载我们自己训练的权重(中文古诗词)
model.load_state_dict(torch.load("params/net.pt"))
#开启测试模式，使用transformers库时，不能直接调用eval,在加载预训练库时，会自动调用eval
#model.eval()

#使用系统自带的Pipeline工具生成内容
pipeline = TextGenerationPipeline(model, tokenizer, device=0)
print(pipeline("天高", max_length=24))