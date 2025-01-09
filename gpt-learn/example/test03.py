#中文文言文生成
from sympy.stats.sampling.sample_pymc import do_sample_pymc
from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline, AutoModel, AutoTokenizer

#加载模型和分词器
model = GPT2LMHeadModel.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-ancient\snapshots\3b264872995b09b5d9873e458f3d03a221c00669")
tokenizer = BertTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-ancient\snapshots\3b264872995b09b5d9873e458f3d03a221c00669")

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model, tokenizer)
#使用text_generator生成文本  续写模型
#do_sample:是否使用采样的方式生成文本,为True时，每次生成的结果都不一样，为False时，每次生成的结果都一样
for i in range(3):
    print(text_generator("于是者", max_length=100, do_sample=True))