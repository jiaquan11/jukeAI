#中文古诗词生成
from sympy.stats.sampling.sample_pymc import do_sample_pymc
from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline, AutoModel, AutoTokenizer

#加载模型和分词器
model = GPT2LMHeadModel.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-poem\snapshots\6335c88ef6a3362dcdf2e988577b7bafeda6052b")
tokenizer = BertTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-poem\snapshots\6335c88ef6a3362dcdf2e988577b7bafeda6052b")

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model, tokenizer)
#使用text_generator生成文本  续写模型
#do_sample:是否使用采样的方式生成文本,为True时，每次生成的结果都不一样，为False时，每次生成的结果都一样
for i in range(3):
    print(text_generator("[CLS]白日依山尽,", max_length=24, do_sample=True))