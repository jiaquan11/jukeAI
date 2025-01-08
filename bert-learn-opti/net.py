from transformers import BertModel
import torch

#定义设备信息
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

#加载预训练模型BERT,并将其加载到设备上,Bert模型是一个特征提取器
pretrained = BertModel.from_pretrained(r'D:\study\computerStudy\personcode\jukeAI\bert-learn-opti\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f').to(DEVICE)
print(pretrained)

#定义下游任务(增量模型)
'''
模型结构：
使用预训练的 BERT 模型作为特征提取器，提取输入文本的语义表示。
在 BERT 的基础上添加一个全连接层，用于完成二分类任务。
前向传播逻辑：
冻结 BERT 模型的参数（torch.no_grad()），只训练全连接层。
输入文本经过 BERT 模型，提取 [CLS] token 的表示。
将 [CLS] token 的表示传入全连接层，得到二分类任务的预测结果。
增量训练：
通过冻结 BERT 的参数，只训练全连接层（fc），实现增量学习。
这种方法适合在小数据集上微调模型，避免过拟合。
'''
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() #构建Bert模型
        #设计全连接网络，实现二分类任务
        #self.fc = torch.nn.Linear(768, 2) #输入维度为768，输出维度为2,二分类任务，用于情感分类
        self.fc = torch.nn.Linear(768, 8)  # 输入维度为768，输出维度为8,多分类任务，用于多分类评价等
    def forward(self, input_ids, attention_mask,token_type_ids):
        #冻结Bert预训练模型的参数，进行前向传播，让其不参与训练，只训练全连接网络，也就是只训练增量模型
        #使用 torch.no_grad() 冻结 BERT 的参数，让它只作为特征提取器，不参与训练
        with torch.no_grad():#表示在这个代码块中不计算梯度，冻结 BERT 模型的参数
            #将生成的 input_ids、attention_mask 和 token_type_ids 传入预训练模型
            #输入的参数为分词器对文本的转换结果，作为预训练模型的输入，这些参数是预训练模型的标准输入格式
            #这里的pretrained只做特征提取，不参与训练
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #增量模型参与训练，输入的是:BERT的最后一层隐藏状态，传入全连接层(fc),进行二分类任务
        #全连接层的输出是一个 未归一化的分数（logits），需要通过激活函数（如 softmax）将其转换为概率分布，然后根据概率的大小判断情感类别
        out = self.fc(out.last_hidden_state[:, 0]) #取出[CLS] token的表示，传入全连接层，得到二分类任务的预测结果,但是这里不能进行分析，因为没有激活函数，只是一个二分类结果
        return out
