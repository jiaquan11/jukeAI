import torch
import numpy as np

#张量运算
#张量运算是PyTorch的核心功能之一，支持大量的张量运算操作
#张量运算包括标量运算、向量运算、矩阵运算等
#标量运算
#加减乘除取负号:
#add, sub, mul, div, neg
#add_, sub_, mul_, div_, neg_(其中带下划线的函数会修改原数据)
data = torch.randint(0, 10, (2, 3))
print(data)
'''
tensor([[0, 7, 0],
        [8, 7, 5]])
'''
#不修改原数据
new_data = data.add(10) #等价于new_data = data + 10
print(new_data)
'''
tensor([[10, 17, 10],
        [18, 17, 15]])
'''
#直接修改原数据，注意：带下划线的函数会修改原数据本身
data.add_(10) #等价于data = data + 10
print(data)
'''
tensor([[10, 17, 10],
        [18, 17, 15]])
'''

#点乘运算
data1 = torch.tensor([[1, 2], [3, 4]])
data2 = torch.tensor([[5, 6], [7, 8]])
#第一种方式
data = torch.mul(data1, data2) #等价于data = data1 * data2
print(data)
'''
tensor([[ 5, 12],
        [21, 32]])
'''

'''
乘法运算
数组乘法运算要求第一个数组shape:(n,m),第二个数组shape:(m,p),两个数组乘法运算shape为(n,p)
运算符@用于进行两个矩阵的乘积运算
torch.matmul中输入的shape不同的张量，对应的维度必须符合数组乘法的运算规则
乘法的规则是：
唯一需要满足的条件是第一个矩阵的列数等于第二个矩阵的行数，具体的数字内容可以完全不同
第一个矩阵的每一行与第二个矩阵的每一列进行点积，计算出结果矩阵的元素
'''
#乘法运算
data_1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
data_2 = torch.tensor([[5, 6], [7, 8]])
#方式1
data_3 = data_1 @ data_2
print("data_3->", data_3)
'''
data_3-> tensor([[19, 22],
        [43, 50],
        [67, 78]])
'''
#方式2
data_4 = torch.matmul(data_1, data_2)
print("data_4->", data_4)
'''
data_4-> tensor([[19, 22],
        [43, 50],
        [67, 78]])
'''