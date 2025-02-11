#定义一个函数
def func(name, age, address):
    print(name)
    print(age)
    print(address)

#1.位置传参
func("张三", 18, "北京")

#2.关键字传参，位置可变
func(name="李四", address="上海", age=20)