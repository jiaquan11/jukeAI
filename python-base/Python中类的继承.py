class Person(object):
    def __init__(self, name, age): #self自动产生的，对象本身
        self.name = name
        self.age = age

    def eat(self):
        print("吃东西")

class Student(Person):
    pass

s1 = Student("李四", 20)
print(s1.name)
print(s1.age)
s1.eat()