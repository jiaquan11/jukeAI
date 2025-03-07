from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 设置具体包含 config.json 的目录
model_dir = r"D:\study\computerStudy\personcode\jukeAI\huggingface-learn\transformers_test\model\uer\gpt2-chinese-cluecorpussmall\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"

#加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

#使用加载的模型和分词器创建生成文本的pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cpu')

#生成文本
#output = generator("你好，我是一款语言模型，", max_length=50, num_return_sequences=1, truncation=True)
# output = generator("你好，我是一款语言模型，", max_length=50, num_return_sequences=1, truncation=True, clean_up_tokenization_spaces=False)
output = generator(
    "你好，我是一款语言模型，",#生成文本的输入种子文本（prompt）。模型会根据这个初始文本，生成后续的文本
    max_length=50,#指定生成文本的最大长度。这里的 50 表示生成的文本最多包含 50 个标记（tokens）
    num_return_sequences=2,#参数指定返回多少个独立生成的文本序列。值为 1 表示只生成并返回一段文本。
    truncation=True,#该参数决定是否截断输入文本以适应模型的最大输入长度。如果 True，超出模型最大输入长度的部分将被截断；如果 False，模型可能无法处理过长的输入，可能会报错。
    temperature=0.7,#该参数控制生成文本的随机性。值越低，生成的文本越保守（倾向于选择概率较高的词）；值越高，生成的文本越多样（倾向于选择更多不同的词）。0.7 是一个较为常见的设置，既保留了部分随机性，又不至于太混乱。
    top_k=50,#该参数限制模型在每一步生成时仅从概率最高的 k 个词中选择下一个词。这里 top_k=50 表示模型在生成每个词时只考虑概率最高的前 50 个候选词，从而减少生成不太可能的词的概率。
    top_p=0.9,#该参数（又称为核采样）进一步限制模型生成时的词汇选择范围。它会选择一组累积概率达到 p 的词汇，模型只会从这个概率集合中采样。top_p=0.9 意味着模型会在可能性最强的 90% 的词中选择下一个词，进一步增加生成的质量。
    clean_up_tokenization_spaces=True#该参数控制生成的文本中是否清理分词时引入的空格。如果设置为 True，生成的文本会清除多余的空格；如果为 False，则保留原样。默认值即将改变为 False，因为它能更好地保留原始文本的格式。
)

print(output)