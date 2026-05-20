def dataclass_decorator(typechekck=False):
    def wrapper(cls):
        params = cls.__annotations__

        def dynamic_init(self, *args):
            for p, a in zip(params, args):
                if typechekck:
                    if not isinstance(a, params[p]):
                        raise TypeError(f'{p} expected type is {params[p].__name__}, not {type(a).__name__}')
                self.__dict__[p] = a

        def dynamic_repr(self):
            return f'{cls.__name__}({", ".join(f'{k}={v}' for k, v in self.__dict__.items())})'
        
        cls.__init__ = dynamic_init
        cls.__repr__ = dynamic_repr
        return cls
    
    if isinstance(typechekck, bool):
        return wrapper
    return wrapper(typechekck)

# @dataclass_decorator
# class Employee:
#     name: str
#     age: int

# e = Employee('Ivan', 20)
# print(e)

# --- class ---

class dataclass:
    def __new__(cls, *_):
        params = cls.__annotations__

        def dynamic_init(self, *args):
            for p, a in zip(params, args):
                self.__dict__[p] = a

        def dynamic_repr(self):
            return f'{cls.__name__}({", ".join(f'{k}={v}' for k, v in self.__dict__.items())})'
        
        cls.__init__ = dynamic_init
        cls.__repr__ = dynamic_repr
        return object.__new__(cls)

class Employee(dataclass):
    name: str
    age: int

e = Employee('Ivan', 20)
print(e)