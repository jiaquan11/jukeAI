class Person(object):
    def __init__(self, name, age): #self自动产生的，对象本身
        self.name = name
        self.age = age

    def eat(self):
        print("吃东西")

p1 = Person("张三", 18)
print(p1.name)
print(p1.age)