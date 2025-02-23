from langchain.text_splitter import CharacterTextSplitter

# 分割器实例化对象
text_spliter = CharacterTextSplitter(separator=' ',
                                     chunk_size=5, #指明每个分割文本块的大小
                                     chunk_overlap=5, #指明每个分割后的文档之间的重复字符个数
                                     )

# 对一句话进行分割
result1 = text_spliter.split_text("a b c d e f")
print(f'一句话分割的结果{result1}')

# 对多个句子也就是文档切分
# texts = text_spliter.create_documents(["a b c d e f", "e f g h"])
# print(f'texts-->{texts}')