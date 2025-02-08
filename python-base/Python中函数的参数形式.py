#1、普通参数
def func1(num1, num2):
    print(num1, num2)

print(func1(10, 20))#10 20

#2、默认参数
def func2(a, b, c = 100):
    return a + b + c

func2(1, 10) #111
func2(1, 10, 200) #211

#3、可变参数
def func3(*args, **kwargs):
    print(args) #元组,接收位置传参
    print(kwargs) #字典,接收关键字传参

func3(10, 20, 30) #位置传参 (10, 20, 30) {}
func3(a = 10, b = 20, c = 30) #关键字传参 () {'a': 10, 'b': 20, 'c': 30}