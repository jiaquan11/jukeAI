import torch
import torch.nn as nn
from torchsummary import summary

#创建一个神经网络模型
class Model(nn.Module):
    #初始化属性值
    def __init__(self):
        super(Model, self).__init__() #调用父类的初始化属性值
        self.l1 = nn.Linear(3, 3) #创建第一个隐藏层模型，3个输入特征，3个输出特征
        self.l2 = nn.Linear(3, 2) #创建第二个隐藏层模型，3个输入特征(上一层的输出特征)，2个输出特征
        self.out = nn.Linear(2, 2) #创建输出层模型，2个输入特征(上一层的输出特征)，2个输出特征

    #创建前向传播函数，自动执行forward()函数
    def forward(self, x):
        #数据经过第一个线性层，并使用sigmoid激活函数
        x = torch.sigmoid(self.l1(x))
        #数据经过第二个线性层，并使用relu激活函数
        x = torch.relu(self.l2(x))
        #数据经过输出层
        x = self.out(x)
        #使用softmax激活函数，dim=-1表示对最后一个维度进行softmax激活
        out = torch.softmax(x, dim=-1)
        return out

if __name__ == '__main__':
    #实例化model对象
    model = Model()
    #随机产生一个5*3的张量数据
    x = torch.randn(5, 3)
    #数据经过神经网络模型训练
    out = model(x)
    #打印输出数据的形状
    print(out.shape)
    #计算模型参数
    #计算每层每个神经元的w和b参数的个数总和
    summary(model, input_size=(3,), batch_size=5)
