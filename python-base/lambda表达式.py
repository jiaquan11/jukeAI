'''
def func1():
    只有一行语句

---------------------
func2 = lambda 参数:返回值
'''

def func1(num1, num2):
    return num1 ** num2 #num1的num2次方

print(func1(10, 20)) #30

func2 = lambda num1, num2: num1 ** num2
print(func2(10, 20)) #30