class Employee:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def a(self):
        return 2

def vars(obj):
    # return obj.__dict__
    params = [p for p in dir(obj) if not p.startswith('__') and not callable(getattr(obj, p))]
    vals = [getattr(obj, p) for p in params]

    return {key: val for key, val in zip(params, vals)}

e = Employee("ivan", 23)
print(dir(e))
data = vars(e)
print(data)
# { "name": "ivant", "age": 23 }