#中文白话文文章生成
from sympy.stats.sampling.sample_pymc import do_sample_pymc
from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline, AutoModel, AutoTokenizer

#加载模型和分词器
model = GPT2LMHeadModel.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
tokenizer = BertTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model, tokenizer)
#使用text_generator生成文本  续写模型
#do_sample:是否使用采样的方式生成文本,为True时，每次生成的结果都不一样，为False时，每次生成的结果都一样
for i in range(3):
    print(text_generator("这是很久之前的事情了", max_length=100, do_sample=True))