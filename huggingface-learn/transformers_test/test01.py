#将模型下载到本地调用
from transformers import AutoModelForCausalLM, AutoTokenizer

#将模型和分词器下载到本地，并指定保存路径，这样就可以在本地调用模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "model/uer/gpt2-chinese-cluecorpussmall"

#下载模型
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
#下载分词工具
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型分词器已下载到: {cache_dir}")