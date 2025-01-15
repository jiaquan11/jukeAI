from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.optimization import get_scheduler
import torch
from urllib3.filepost import writer

from data import MyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorboardX import SummaryWriter

#实例化数据集
dataset = MyDataset()

#加载编码器
tokenizer = AutoTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

#文本分词编码
def collate_fn(data):
    data = tokenizer.batch_encode_plus(
        data,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt')

    data['labels'] = data['input_ids'].clone()
    return data

#加载数据集
loader = DataLoader(
    dataset=dataset,
    batch_size=6,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
)
print(f"数据的长度:{len(loader)}")
#创建Tensor SummerWriter实例
writer = SummaryWriter("logdir/")

#训练
def train(scaler):
    #全局变量
    global model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)
    model = model.to(DEVICE)

    #定义优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    #定义学习率调度器,下面是以线性方式调整学习率，使优化器在调整过程中更加平滑，稳定
    #训练时长比较长，难度比较大的任务，可以使用这种方式
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(loader))

    model.train() #一般默认是训练模式，不需要调用

    for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to(DEVICE)
        #使用autocast来自动处理混合精度训练
        with autocast():
            out = model(**data) #前向传播
            loss = out['loss'] #获取损失
        #使用梯度缩放器来缩放损失，并调用反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        #out = model(**data)  # 前向传播
        #loss = out['loss']  # 获取损失
        #loss.backward()#反向传播
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)#梯度裁剪
        #optimizer.step()#更新参数
        #scheduler.step()#更新学习率

        #optimizer.zero_grad()#梯度清零
        #model.zero_grad()#模型梯度清零

        #每50个批次输出一次训练信息
        if i % 50 == 0:
            labels = data['labels'][:,1:]#准备标签和输出用于计算准确率
            out = out['logits'].argmax(dim=2)[:,:-1]#通过logits获取模型的原始输出值

            #移除在数据预处理阶段添加的填充(通常是0)，以便只计算实际数据部分的损失和准确率，避免填充部分对模型性能评估的影响
            select = labels != 0
            labels = labels[select]
            out = out[select]
            del select

            accuracy = (labels == out).sum().item() / labels.numel() #输入和模型输出的相似度比较
            lr = optimizer.state_dict()['param_groups'][0]['lr'] #获取当前学习率 学习率如果在减少，说明模型在逐步收敛，稳定
            #相似度只能作为参考，不能作为最终的评价指标，主要是看损失值，看是否在下降
            print(i, loss.item(), lr, accuracy)

            #将训练过程中的损失值、学习率和准确率写入TensorBoard
            writer.add_scalar("Loss/Train", loss.item(), epoch * len(loader) + i)
            writer.add_scalar("acc/Train", accuracy, epoch * len(loader) + i)

    #保存最后一轮模型的参数，生成模型与编码模型Bert不一样，不能选择最优参数，因为模型是生成模型，不是分类模型
    #生成模型的客观评价指标是不全面的
    torch.save(model.state_dict(), "params/net.pt")
    print("权重保存成功")

if __name__ == "__main__":
    # 初始化梯度缩放器
    scaler = torch.amp.GradScaler('cuda')
    #进行1000个训练周期
    for epoch in range(1000):
        #调用训练函数
        train(scaler)
