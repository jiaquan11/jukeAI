#1.定义一个空集合
set1 = set()

#2.定义一个有数据集合
set2 = {1, 11, 22, 8, 9, 10}
print(set2)
print(type(set2))

#3.无序且天生去重
set3 = {1, 11, 22, 8, 9, 10, 8, 10, 11}
print(set3)

set11 = {1, 2, 3, 4, 5}
set22 = {4, 5, 6, 7, 8}
#4.集合的交集
print(set11 & set22)
#5.集合的并集
print(set11 | set22)
#6.集合的差集
print(set11 - set22)