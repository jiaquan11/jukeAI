#模型训练
#加载数据和模型，将数据输入模型进行训练，得到输出，根据输出计算损失，把损失丢入优化器优化模型参数,反向传播，更新参数

import torch
#from MyData import MyDataset
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer, AdamW

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义训练的轮次
EPOCH = 30000

#从模型中加载分词器
token = BertTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\datasets-opti\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

def collate_fn(data):
    '''
    数据集的处理函数
    '''
    #获取数据集中的文本和标签
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    #对文本进行编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
        return_length=True,
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label) #将标签转换为张量
    return input_ids, attention_mask, token_type_ids, labels

#创建训练数据集
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=100,#批次大小，每次训练的数据量
    shuffle=True,#是否打乱数据
    drop_last=True,#是否丢弃最后一个不足一个批次的数据,防止形状出错
    #对加载进来的数据进行编码
    collate_fn=collate_fn
)

val_dataset = MyDataset("validation")
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=100,#批次大小，每次训练的数据量
    shuffle=True,#是否打乱数据
    drop_last=True,#是否丢弃最后一个不足一个批次的数据,防止形状出错
    #对加载进来的数据进行编码
    collate_fn=collate_fn
)

if __name__ == '__main__':
    #开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    #定义优化器
    optimizer = AdamW(model.parameters()) #有个默认的学习率，暂时不设置
    #定义损失函数
    #使用交叉熵损失函数（CrossEntropyLoss）来优化模型：
    # 输入：模型的输出 logits 和真实标签。
    #目标：最小化预测类别与真实类别之间的差距
    loss_func = torch.nn.CrossEntropyLoss()

    #初始化最佳验证准确率
    best_val_acc = 0.0

    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            #将数据存放到DEVICE中
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            #前向计算(将分词编码后的数据输入模型，得到输出)，out 是未归一化的分数（logits），表示模型对每个类别的预测概率分数(小数)
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            #根据二分类预测输出结果，与训练标签计算损失
            loss = loss_func(out, labels)
            #根据损失，优化参数
            optimizer.zero_grad() #梯度清零
            loss.backward() #反向传播
            optimizer.step() #更新参数

            #每隔5个批次输出训练信息
            if i%5 == 0:
                out = out.argmax(dim=1) #通过argmax函数从模型的输出中选出每个样本的预测类别,非0即1
                acc = (out == labels).sum().item() / len(labels) #计算准确率，即精度
                print(f"epoch:{epoch}, i:{i}, loss:{loss.item()}, acc:{acc}")

        #验证模型(判断是否过拟合) (训练集训练完一轮后，用验证集验证模型)
        #设置为评估模式
        model.eval()
        #不需要模型参与训练
        with torch.no_grad():
            val_acc = 0.0
            val_loss = 0.0
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
                # 将数据存放到DEVICE中
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
                # 前向计算(将分词编码后的数据输入模型，得到输出)，out 是未归一化的分数（logits），表示模型对每个类别的预测概率分数(小数)
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # 根据二分类预测输出结果，与训练标签计算损失
                val_loss += loss_func(out, labels)
                out = out.argmax(dim=1)
                val_acc += (out == labels).sum().item()
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f"验证集 loss:{val_loss}, acc:{val_acc}")

            #根据验证集的准确率保存最优模型，根据每一轮验证集的准确率保存最优模型
            #过拟合没法解决，只能防止，就是只能保存最优模型参数，不保存过拟合的参数
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "params/news_best_bert.pth")
                print(f"Epoch:{epoch},保存最优参数 acc:{best_val_acc}")

        #保存最后一轮的训练参数
        torch.save(model.state_dict(), f"params/news_last_bert.pth")
        print(epoch, f"Epoch:{epoch},最后一轮参数保存成功！")

