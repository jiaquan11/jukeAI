#导入工具包
import torch
from torch.utils.data import TensorDataset #构造数据集对象
from torch.utils.data import DataLoader #构造数据加载器对象
from torch import nn #nn模块中有平方损失函数和假设函数
from torch import optim #optim模块中有优化器函数
from sklearn.datasets import make_regression #创建线性回归模型的数据集
import matplotlib.pyplot as plt #绘图工具包

plt.rcParams['font.sans-serif'] = ['SimHei'] #设置正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #设置正常显示负号

#构建数据集
def create_dataset():
    x,y,coef = make_regression(n_samples=100,
                               n_features=1,
                               noise=10,
                               coef=True,
                               bias=1.5,
                               random_state=0)
    #将构建数据转换张量类型
    x = torch.tensor(x)
    y = torch.tensor(y)
    return x,y,coef

if __name__ == "__main__":
    #生成的数据
    x,y,coef = create_dataset()

    # #绘制数据的真实的线性回归结果
    # plt.scatter(x,y)
    # x = torch.linspace(x.min(), x.max(), 1000)
    # y1 = torch.tensor([v * coef + 1.5 for v in x])
    # plt.plot(x,y1, label='real')
    # plt.grid()
    # plt.legend()
    # plt.show()

    dataset = TensorDataset(x, y) #构造数据集对象
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True) #构造数据加载器对象
    model = nn.Linear(in_features=1, out_features=1) #构造线性回归模型
    #损失和优化器
    loss = nn.MSELoss() #构造均方差损失函数
    optimizer = optim.SGD(params=model.parameters(), lr=0.01) #构造随机梯度下降优化器

    #训练模型
    epochs = 100
    #损失的变化
    loss_epoch = []
    total_loss = 0.0
    train_sample = 0.0
    for _ in range(epochs):
        for train_x, train_y in dataloader:
            #将一个batch的训练数据送入模型
            y_pred = model(train_x.type(torch.float32))
            #计算损失
            loss_values = loss(y_pred, train_y.reshape(-1, 1).type(torch.float32))
            total_loss += loss_values.item()
            train_sample += len(train_y)
            #梯度清零
            optimizer.zero_grad()
            #自动微分(反向传播)
            loss_values.backward()
            #更新参数
            optimizer.step()
        #获取每个batch的平均损失
        loss_epoch.append(total_loss / train_sample)


    #绘制损失变化曲线
    plt.plot(range(epochs), loss_epoch)
    plt.show()

    #绘制预测和真实的线性回归拟合结果
    plt.scatter(x,y)
    x1 = torch.linspace(x.min(), x.max(), 1000)
    y0 = torch.tensor([v*model.weight+model.bias for v in x1]) #根据特征学习的预测值
    y1 = torch.tensor([v * coef + 1.5 for v in x1]) #真实值
    plt.plot(x1,y0, label='预测')
    plt.plot(x1,y1, label='真实')
    plt.legend()
    plt.grid()
    plt.show()