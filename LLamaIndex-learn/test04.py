from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings,SimpleDirectoryReader,VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SimpleNodeParser

#初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbedding(
    #指定了一个预训练的sentence-transformer模型的路径
    model_name="/root/public/llm/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

#将创建的嵌入模型赋值给全局设置的embed_model属性，这样在后续的索引构建过程中，就会使用这个模型
Settings.embed_model = embed_model

#使用HuggingFaceLLM加载本地大模型
llm = HuggingFaceLLM(model_name="/root/public/llm/Qwen/Qwen1___5-1___8B-Chat",
               tokenizer_name="/root/public/llm/Qwen/Qwen1___5-1___8B-Chat",
               model_kwargs={"trust_remote_code":True},
               tokenizer_kwargs={"trust_remote_code":True})
#设置全局的llm属性，这样在索引查询时会使用这个模型。
Settings.llm = llm

#从指定目录读取文档，将数据加载到内存
documents = SimpleDirectoryReader("/root/public/projects/demo_22/data").load_data()
# print(documents)
#创建节点解析器
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
#将文档分割成节点
base_node = node_parser.get_nodes_from_documents(documents=documents)
print(documents)
print("=================================")
print(base_node)
print("=================================")
#根据自定义的node节点构建向量索引
index = VectorStoreIndex(nodes=base_node)

#创建一个VectorStoreIndex,并使用之前加载的文档来构建向量索引
#此索引将文档转换为向量，并存储这些向量（内存）以便于快速检索
# index = VectorStoreIndex.from_documents(documents)

#将索引持久化存储到本地的向量数据库
index.storage_context.persist()

#创建一个查询引擎，这个引擎可以接收查询并返回相关文档的响应。
query_engine = index.as_query_engine()
rsp = query_engine.query("xtuner的使用步骤是什么？")
print(rsp)