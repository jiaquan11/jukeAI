from datasets import load_dataset,load_from_disk

#在线加载数据
#dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",split="train")
#print(dataset)

#加载本地磁盘数据
dataset = load_from_disk(r"D:\study\computerStudy\personcode\jukeAI\huggingface-learn\data\ChnSentiCorp")
print(dataset)

#取出测试集
test_data = dataset["test"]
print(test_data)
#查看数据
for data in test_data:
    print(data)