#模型训练
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer,AdamW

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载字典和分词器
token = BertTokenizer.from_pretrained(r"D:\PycharmProjects\demo_02\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

#将传入的字符串进行编码
def collate_fn(data):
    sents = [i[0]for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        # 当句子长度大于max_length(上限是model_max_length)时，截断
        truncation=True,
        max_length=512,
        # 一律补0到max_length
        padding="max_length",
        # 可取值为tf,pt,np,默认为list
        return_tensors="pt",
        # 返回序列长度
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    label = torch.LongTensor(label)
    return input_ids,attention_mask,token_type_ids,label

#创建数据集
test_dataset = MyDataset("test")
test_loader = DataLoader(
    dataset=test_dataset,
    #训练批次
    batch_size=80,
    #打乱数据集
    shuffle=True,
    #舍弃最后一个批次的数据，防止形状出错
    drop_last=True,
    #对加载的数据进行编码
    collate_fn=collate_fn
)

if __name__ == '__main__':
    acc = 0.0
    total = 0

    #开始测试
    print(DEVICE)
    model = Model().to(DEVICE)
    #加载模型训练参数
    model.load_state_dict(torch.load("params/16_bert.pth"))
    #开启测试模式
    model.eval()

    for i,(input_ids,attention_mask,token_type_ids,label) in enumerate(test_loader):
        #将数据放到DVEVICE上面
        input_ids, attention_mask, token_type_ids, label = input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),label.to(DEVICE)
        #前向计算（将数据输入模型得到输出）
        out = model(input_ids,attention_mask,token_type_ids)
        out = out.argmax(dim=1) #one-hot编码,将输出的概率最大的位置作为预测结果，返回索引，这样就可以和标签进行比较
        acc += (out==label).sum().item()
        print(i,(out==label).sum().item())
        total+=len(label)
    print(f"test acc:{acc/total}")