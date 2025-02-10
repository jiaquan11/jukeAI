import torch
import numpy as np

#张量转换为NumPy数组
#使用Tensor.numpy()函数可以将张量转换为numpy数组
#将张量转换为numpy数组
data_tensor = torch.tensor([10, 20, 30])
#使用Tensor.numpy()函数将张量转换为numpy数组
data_numpy = data_tensor.numpy()
print(type(data_numpy)) #<class 'numpy.ndarray'>
print(type(data_tensor)) #<class 'torch.Tensor'>

#NumPy数组转换为张量
#使用torch.from_numpy()函数可以将numpy数组转换为张量
#将numpy数组转换为张量
data_numpy = np.array([10, 20, 30])
data_tensor = torch.from_numpy(data_numpy)
print(type(data_numpy)) #<class 'numpy.ndarray'>
print(type(data_tensor)) #<class 'torch.Tensor'>
#注意：转换后的张量与原numpy数组共享内存，修改一个会影响另一个

#使用torch.tensor()可以将numpy数组转换为张量，但不共享内存
data_numpy = np.array([10, 20, 30])
data_tensor = torch.tensor(data_numpy)
print(data_tensor)
print(data_numpy)
'''
tensor([10, 20, 30])
'''

#标量张量和数字之间的转换
#对于只有一个元素的张量，可以使用item()函数将该值从张量中提取出来
data = torch.tensor([30,])
print(data.item()) #30
data = torch.tensor(30)
print(data.item()) #30