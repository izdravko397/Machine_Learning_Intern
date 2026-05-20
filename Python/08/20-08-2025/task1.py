def singleton(cls):
    cls.first_instance = None
    def __new__(cls):
        if not cls.first_instance:
            cls.first_instance = object.__new__(cls)

        return cls.first_instance
    
    cls.__new__ = __new__
    return cls

@singleton
class MyClass:
    pass
  
mc1 = MyClass()
mc2 = MyClass()

print(mc1 is mc2)
# True