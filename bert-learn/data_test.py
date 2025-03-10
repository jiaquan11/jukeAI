from datasets import load_dataset, load_from_disk

#huggingface的数据集是特有的格式，比如arrow格式，所以不能直接使用pandas读取
#但是可以使用datasets库的load_dataset方法加载数据集，然后转存为CSV格式，再使用pandas读取
#huuggingface的加载数据集的接口load_dataset也可以直接加载CSV格式的数据集

#在线加载数据集 (hermes-function-calling-v1数据集，用于函数调用)  (不管本地有没有缓存，都要科学上网，进行下载)
#dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",split="train")
#print(dataset)

#转存为CSV格式
#dataset.to_csv(path_or_buf=r"D:\study\computerStudy\personcode\jukeAI\bert-learn\data\hermes-function-calling-v1.csv")

#加载CSV格式数据集(也是本地),  注意:load_from_disk只能加载缓存目录的数据集，不能加载CSV文件，所以这里不能使用load_from_disk
#dataset = load_dataset(path="csv", data_files=r"D:\study\computerStudy\personcode\jukeAI\bert-learn\data\hermes-function-calling-v1.csv")
#print(dataset)

#加载缓存目录数据集,情感分析数据集，ChnSentiCorp
dataset = load_from_disk(r"D:\study\computerStudy\personcode\jukeAI\bert-learn\data\ChnSentiCorp")
print(dataset)

test_data = dataset["test"]
for data in test_data:
    print(data)