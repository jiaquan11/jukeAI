class Adder(object):
    def __init__(self, value = 0):
        self.data = value

    #当一个类的实例被调用时，就会自动触发__call__()方法
    def __call__(self, x):
        return self.data + x

adder = Adder()
print(adder(10)) #10
print(adder(20)) #20