import torch

#自动微分模块
#1.当X为标量时梯度的计算
def test01():
    x = torch.tensor(5)
    #目标值
    y = torch.tensor(0.)
    #设置要更新的权重和偏置的初始值
    w = torch.tensor(1., requires_grad=True,dtype=torch.float32)
    b = torch.tensor(3., requires_grad=True,dtype=torch.float32)
    #设置网络的输出值
    z = w * x + b #矩阵乘法
    #设置损失函数，并进行损失的计算
    loss = torch.nn.MSELoss()
    loss = loss(z, y)
    #自动微分
    loss.backward()
    #打印w,b变量的梯度
    #backward()函数会自动计算梯度，梯度值保存在grad属性中
    print('W的梯度: ', w.grad)
    print('b的梯度：', b.grad)

def test02():
    #输入张量2*5
    x = torch.ones(2,5)
    #目标值是2*3
    y = torch.zeros(2,3)
    #设置要更新的权重和偏置的初始值
    w = torch.randn(5,3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    #设置网络的输出值
    z = torch.matmul(x, w) + b #矩阵乘法
    #设置损失函数，并进行损失的计算
    loss = torch.nn.MSELoss()
    loss = loss(z, y)
    #自动微分
    loss.backward()
    #打印w,b变量的梯度
    #backward()函数会自动计算梯度，梯度值保存在grad属性中
    print('W的梯度: ', w.grad)
    print('b的梯度：', b.grad)




test01()