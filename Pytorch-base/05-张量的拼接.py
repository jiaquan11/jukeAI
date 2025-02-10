import torch

#张量拼接
#使用torch.cat()函数可以将多个张量拼接在一起
#cat()函数可以将张量按照指定的维度拼接在一起
data1 = torch.randint(0, 10, [1, 2, 3])
data2 = torch.randint(0, 10, [1, 2, 3])
#1.按0维度拼接
new_data = torch.cat([data1, data2], dim=0)
print(new_data.shape) #torch.Size([2, 2, 3])

#2.按1维度拼接
new_data = torch.cat([data1, data2], dim=1)
print(new_data.shape) #torch.Size([1, 4, 3])

#3.按2维度拼接
new_data = torch.cat([data1, data2], dim=2)
print(new_data.shape) #torch.Size([1, 2, 6])