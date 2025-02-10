'''
PyTorch中的张量运算函数
均值，平方根，求和，指数计算，对数计算，三角函数，矩阵运算，排序，统计函数等
'''
import torch
import numpy as np

data = torch.randint(0, 10, [2, 3], dtype = torch.float64)
print(data)

#计算均值
#注意:tensor必须为Float或者Double类型
print(data.mean()) #tensor(4.3333, dtype=torch.float64)

#计算总和
print(data.sum()) #tensor(26., dtype=torch.float64)

#计算平方
print(torch.pow(data, 2)) #tensor([[ 9., 16.,  1.],

#计算平方根
print(data.sqrt()) #tensor([[1.7321, 2.4495, 1.0000],

#计算指数,e为底,计算每个元素的指数
print(data.exp()) #tensor([[  2.7183, 1096.6332,   2.7183],

#计算对数,计算每个元素的对数
print(data.log()) #tensor([[0.0000, 1.9459, 0.0000],
print(data.log2()) #tensor([[0.0000, 2.0000, 0.0000],
print(data.log10()) #tensor([[0.0000, 0.7782, 0.0000],