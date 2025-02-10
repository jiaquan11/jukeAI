import torch
import numpy as np
'''
掌握张量形状操作的方法
reshape(), squeeze(), unsqueeze(), transpose(), permute(),
view(),contiguous(), expand(), repeat(), tile()等函数的使用
'''

#reshape()函数
#使用reshape()函数可以在保证张量数据不变的情况下改变数据的维度，将其转换为指定形状
data = torch.tensor([[10, 20, 30], [40, 50, 60]])
#1、使用shape属性或者size()函数查看张量的形状
print(data.shape) #torch.Size([2, 3])
print(data.size()) #torch.Size([2, 3])

#2、使用reshape()函数将张量转换为指定形状
data_new = data.reshape(1, 6)
print(data_new.shape) #torch.Size([1, 6])

#sequeeze()函数和unsqueeze()函数
#sequeeze()函数用于删除维度为1的维度，unsqueeze()函数用于在指定位置增加维度为1的维度
mydata1 = torch.tensor([1, 2, 3, 4, 5])
print('mydata1->', mydata1.shape,mydata1)
'''
mydata1-> torch.Size([5]) tensor([1, 2, 3, 4, 5])
'''
mydata2 = mydata1.unsqueeze(dim=0)
print('在0维度上拓展维度:', mydata2.shape, mydata2)
'''
在0维度上拓展维度: torch.Size([1, 5]) tensor([[1, 2, 3, 4, 5]])
'''
mydata3 = mydata1.unsqueeze(dim=1)
print('在1维度上拓展维度:', mydata3.shape, mydata3)
'''
在1维度上拓展维度: torch.Size([5, 1]) tensor([[1],
        [2],
        [3],
        [4],
        [5]])
'''
mydata4 = mydata3.unsqueeze(dim=-1)
print('在-1维度上拓展维度:', mydata4.shape, mydata4)
'''
在-1维度上拓展维度: torch.Size([5, 1, 1]) tensor([[[1]],

        [[2]],

        [[3]],

        [[4]],

        [[5]]])
'''
mydata5 = mydata4.squeeze()
print('删除维度为1的维度:', mydata5.shape, mydata5)
'''
删除维度为1的维度: torch.Size([5]) tensor([1, 2, 3, 4, 5])
'''

#transpose()函数和permute()函数
#transpose()函数用于交换张量形状的指定维度，例如一个张量的形状为(2, 3, 4)，可以使用transpose(1, 2)将其转换为(2, 4, 3)
# permute()函数用于交换张量的多个维度

data = torch.tensor(np.random.randint(0, 10, [3, 4, 5]))
print(data.shape) #torch.Size([3, 4, 5])
#交换1和2维度
mydata2 = torch.transpose(data, 1, 2)
print(mydata2.shape) #torch.Size([3, 5, 4])
#将data的形状修改为(4,5,3),需要变换多次
mydata3 = torch.transpose(data, 0, 1)
mydata4 = torch.transpose(mydata3, 1, 2)
print(mydata4.shape) #torch.Size([4, 5, 3])

#使用permute()函数将形状修改为(4, 5, 3)
mydata5 = torch.permute(data, [1, 2, 0])
print(mydata5.shape) #torch.Size([4, 5, 3])
mydata6 = data.permute([1, 2, 0])
print(mydata6.shape) #torch.Size([4, 5, 3])

'''
view()函数和contiguous()函数
view()函数也可以用于修改张量的形状，只能用于存储在整块内存中的张量。在PyTorch中，张量的存储方式有两种，一种是连续存储，另一种是不连续存储。
有些张量是由不同的数据块组成的，它们并没有存储在整块的内存中，view函数无法对这样的张量进行变形处理
'''
#若要使用view()函数，需要先使用contiguous()函数将张量转换为连续存储，然后再使用view()函数
#判断张量是否使用整块内存存储
data = torch.tensor([[10, 20, 30], [40, 50, 60]])
print('data->', data, data.shape)
#判断是否用整块内存
print(data.is_contiguous()) #True
#使用view()函数修改形状
mydata2 = data.view(3, 2)
print('mydata2->', mydata2, mydata2.shape)
'''
data-> tensor([[10, 20, 30],
        [40, 50, 60]]) torch.Size([2, 3])
mydata2-> tensor([[10, 20],
        [30, 40],
        [50, 60]]) torch.Size([3, 2])
'''