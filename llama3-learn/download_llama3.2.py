
#模型下载
#from modelscope import snapshot_download
#model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct',
#                             cache_dir="model/")

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('UnicomAI/Unichat-llama3.2-Chinese-1B',
                              cache_dir="model/")


#数据集下载
# from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('w10442005/ruozhiba_qa',
#                      subset_name='default',
#                      split='train',
#                      cache_dir="data/")


#将下载的数据集arrow文件转换为json文件
# from datasets import Dataset
# import json
#
# # 加载 .arrow 文件
# dataset = Dataset.from_file(r"D:\study\computerStudy\personcode\jukeAI\llama3-learn\data\w10442005___ruozhiba_qa\default-de7913bb979851d5\0.0.0\master\ruozhiba_qa-train.arrow")
#
# # 定义 JSON 输出格式
# output_data = []
# for example in dataset:
#     # 假设 .arrow 文件中包含 "instruction"、"input" 和 "output" 字段
#     # 如果字段名不同，请根据实际字段修改
#     output_data.append({
#         "instruction": example.get("query", ""),  # 获取 instruction 字段
#         "input": example.get("system", ""),              # 获取 input 字段
#         "output": example.get("response", "")             # 获取 output 字段
#     })
#
# # 将数据保存为 JSON 文件
# output_file = "ruozhiba_qa-train.json"
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(output_data, f, ensure_ascii=False, indent=4)
#
# print(f"JSON 文件已保存到 {output_file}")