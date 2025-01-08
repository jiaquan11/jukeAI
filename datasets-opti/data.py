from datasets import load_dataset

data = load_dataset(path="csv", data_files="data/news/train.csv", split="train")
print(data)

for i in data:
    print(data['text'])