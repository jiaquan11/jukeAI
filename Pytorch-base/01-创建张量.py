#torch.tensor()根据指定的数据创建张量
import torch
import numpy as np
#1、创建一个张量标量
data = torch.tensor(10)
print(data) #tensor(10)

#2、numpy数组，由于data为float64,下面代码也使用该类型
data = np.random.randn(2, 3)
data = torch.tensor(data)
print(data)
'''
tensor([[ 0.0733, -0.0734, -0.0733],
        [ 0.0733, -0.0733,  0.0733]], dtype=torch.float64)
'''

#3、列表，下面代码使用默认元素类型float32
data = [[10., 20., 30.], [40., 50., 60.]]
data = torch.tensor(data)
print(data)
'''
tensor([[10., 20., 30.],
        [40., 50., 60.]])
'''

#torch.Tensor()根据指定形状创建张量，也可以用来创建指定数据的张量
#创建一个形状为(2, 3)的张量，默认dtype为float32
data = torch.Tensor(2, 3)
print(data)
''' 
tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''
#注意:如果传递列表，则创建包含指定元素的张量
data = torch.Tensor([10])
print(data) #tensor([10.])
data = torch.Tensor([[10, 20]])
print(data)
'''
tensor([[10., 20.]])
'''

#创建线性和随机张量
#torch.arange(),torch.linspace()创建线性张量
#在指定区间按照指定步长生成元素[start, end, step]
data = torch.arange(0, 10, 2)
print(data) #tensor([0, 2, 4, 6, 8])

#在指定区间生成指定个数的元素[start, end, num]
data = torch.linspace(0, 11, 10)
print(data) #tensor([ 0.0000,  1.2222,  2.4444,  3.6667,  4.8889,  6.1111,  7.3333,  8.5556,  9.7778, 11.0000])

#torch.randn()创建随机张量
#创建一个形状为(2, 3)的张量，元素服从标准正态分布
data = torch.randn(2, 3)
print(data)
'''
tensor([[-0.3792, -0.0054, -0.2571],
        [ 0.3122,  0.4371, -0.4482]])
'''

#创建0张量和1张量，指定值张量
#torch.zeros()创建0张量
data = torch.zeros(2, 3)
print(data)
'''
tensor([[0., 0., 0.],
        [0., 0., 0.]])
'''

#torch.ones()创建1张量
data = torch.ones(2, 3)
print(data)
'''
tensor([[1., 1., 1.],
        [1., 1., 1.]])
'''

#torch.full()创建指定值张量
data = torch.full((2, 3), 100)
print(data)
'''
tensor([[100, 100, 100],
        [100, 100, 100]])
'''

'''
张量的数据类型转换
data.type(torch.DoubleTensor) #转换为DoubleTensor类型
data.double() #转换为DoubleTensor类型
'''
data = torch.full((2, 3), 100)
print(data.dtype)
#将data元素类型转换为float64类型
data = data.type(torch.DoubleTensor)
print(data.dtype)
#转换为其他类型
#data = data.type(torch.FloatTensor)
#data = data.type(torch.IntTensor)
#data = data.type(torch.LongTensor)