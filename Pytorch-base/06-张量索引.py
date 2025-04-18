import torch

#随机生成数据
data = torch.randint(0, 10, [4, 5])
print(data)
'''
tensor([[8, 6, 6, 1, 3],
        [8, 9, 5, 2, 8],
        [8, 0, 8, 4, 7],
        [2, 6, 6, 9, 9]])
'''

#简单行，列索引
print(data[0]) #tensor([8, 6, 6, 1, 3])
print(data[:, 0]) #tensor([8, 8, 8, 2])

#列表索引
#返回(0,1),(1,2)两个位置的元素
print(data[[0, 1], [1, 2]]) #tensor([6, 5])

#返回0,1行的1,2列共4个元素
print(data[[[0],[1]],[1,2]])
'''
tensor([[6, 6],
        [9, 5]])
'''

#范围索引
#前3行的前2列数据
print(data[:3, :2])
'''
tensor([[8, 6],
        [8, 9],
        [8, 0]])
'''
#第2行到最后的前2列数据
print(data[2:, :2])

#多维索引
data_new = torch.randint(0, 10, [3, 4, 5])
print(data_new)
'''
tensor([[[5, 2, 5, 1, 0],
         [0, 2, 6, 1, 6],
         [2, 1, 9, 2, 7],
         [4, 7, 7, 2, 5]],

        [[1, 0, 6, 8, 2],
         [8, 8, 1, 9, 3],
         [3, 4, 1, 9, 3],
         [3, 2, 9, 5, 3]],

        [[2, 6, 3, 0, 8],
         [2, 7, 0, 7, 5],
         [9, 0, 3, 7, 4],
         [6, 8, 6, 7, 3]]])
'''

#获取0轴上的第一个数据
print(data_new[0,:,:])
'''
tensor([[5, 2, 5, 1, 0],
        [0, 2, 6, 1, 6],
        [2, 1, 9, 2, 7],
        [4, 7, 7, 2, 5]])
'''
#获取1轴上的第一个数据
print(data_new[:,0,:])
'''
tensor([[5, 2, 5, 1, 0],
        [1, 0, 6, 8, 2],
        [2, 6, 3, 0, 8]])
'''
#获取2轴上的第一个数据
print(data_new[:,:,0])
'''
tensor([[5, 0, 2, 4],
        [1, 8, 3, 3],
        [2, 2, 9, 6]])
'''
