import time

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

#构建数据集
def create_dataset():
    #使用pandas读取数据
    data = pd.read_csv('data/手机价格预测.csv')
    #特征值和目标值
    x,y = data.iloc[:,:-1],data.iloc[:,-1]
    #类型转换:特征值，目标值
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    #数据集划分
    x_train,x_valid,y_train,y_valid=train_test_split(x,y,train_size=0.8,random_state=88)
    #构建数据集，转换为pytorch的形式
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.tensor(y_train.values))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid.values), torch.tensor(y_valid.values))
    #返回结果
    return train_dataset,valid_dataset,x_train.shape[1],len(np.unique(y))

#构建网络模型
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(PhonePriceModel, self).__init__()
        #1.第一层:输入为维度为20，输出为维度为128
        self.fc1 = nn.Linear(input_dim, 128)
        #2.第二层:输入为维度为128，输出为维度为256
        self.fc2 = nn.Linear(128, 256)
        #3.第三层:输入为维度为256，输出为维度为4
        self.fc3 = nn.Linear(256, out_dim)

    def forward(self, x):
        #前向传播过程
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        #获取数据结果
        return output

#训练模型
def train(train_dataset, input_dim, class_num,):
    #初始化模型
    model = PhonePriceModel(input_dim, class_num)
    #损失函数
    loss = nn.CrossEntropyLoss()
    #优化方法
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    #训练轮数
    num_epochs = 50
    #遍历每个轮次的数据
    for epoch_idx in range(num_epochs):
        #初始化数据加载器
        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        #训练时间
        start = time.time()
        #计算损失
        total_loss = 0.0
        total_num = 1
        #遍历每个batch的数据进行处理
        for x, y in dataloader:
            #将数据送入网络中进行预测
            predict = model(x)
            #计算损失
            loss_values = loss(predict, y)
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            loss_values.backward()
            #更新参数
            optimizer.step()
            #计算损失
            total_loss += loss_values.item() #统计损失
            total_num += 1 #统计样本数
        #打印损失变换结果
        print('Epoch: %4s, Loss: %.2f, Time: %.2fs' % (epoch_idx+1, total_loss / total_num, time.time() - start))
    #模型保存
    torch.save(model.state_dict(), 'model/phone_price_model.pth')

#模型评估
def test(valid_dataset, input_dim, class_num):
    #加载模型和训练好的网络参数
    model = PhonePriceModel(input_dim, class_num)
    model.load_state_dict(torch.load('model/phone_price_model.pth'))
    # 初始化数据加载器
    dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8)
    #评估测试集
    correct = 0
    #设置模型为评估模式
    model.eval()
    #遍历每个batch的数据
    for x, y in dataloader:
        #模型预测
        predict = model(x)
        #获取类别结果
        y_pred = torch.argmax(predict, dim=1)
        #计算预测正确的个数
        correct += (y_pred == y).sum()
    #求预测精度
    print('Accuracy: %.5f' % (correct.item() / len(valid_dataset)))

if __name__ == '__main__':
    #获取数据
    train_dataset,valid_dataset,input_dim,class_num = create_dataset()
    print("输入特征数:", input_dim)
    print("分类数:", class_num)

    # #模型实例化
    # model = PhonePriceModel(input_dim, class_num)
    # #统计模型参数
    # summary(model, input_size=(input_dim,), batch_size=16)

    #训练模型
    train(train_dataset, input_dim, class_num)

    #模型评估
    test(valid_dataset, input_dim, class_num)