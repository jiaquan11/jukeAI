from transformers import pipeline,AutoTokenizer,AutoModelWithLMHead

#指明预训练模型的名称
#model_name = 'liam168/trans-opus-mt-zh-en'
model_name = r'D:\study\computerStudy\personcode\jukeAI\Pytorch-base\model\trans-opus-mt-zh-en'
#加载预训练模型
model = AutoModelWithLMHead.from_pretrained(model_name)
#加载词嵌入器
tokenizer = AutoTokenizer.from_pretrained(model_name)
#使用管道的方式进行机器翻译
translator = pipeline('translation_zh_to_en', model=model, tokenizer=tokenizer)
#将要翻译的文本传递到API中
out = translator("我是一个好学生")
print(out)