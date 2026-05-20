def strings(*args):
    res = []

    is_collection = lambda x: isinstance(x, (list, tuple, set, dict))
    is_str = lambda x: isinstance(x, str)
    is_dynamic_obj = lambda x: hasattr(x, '__dict__')

    def seeker(collection):
        for item in collection:
            if is_collection(item):
                seeker(item)

            if is_str(item):
                res.append(item)

            if is_dynamic_obj(item):
                for v in item.__dict__.values():
                    seeker((v,))

            if isinstance(collection, dict):
                seeker((collection[item],))
            
    seeker(args)
    return res

a = ["ivan", 3, 4, 5, None]
s = strings(a) # ["ivan"]
print(s)

class Employee:
    def __init__(self, name):
        self.name = name
        self.a = 1
        self.collect = ['mitko']

e = Employee("Dragan")
a = ["ivan", 3, 4, 5, e, None]
s = strings(a) # ["ivan", "Dragan"]
print(s)

a = [3, 4, 5, None, {"eva", 3}, "nikola"]
s = strings(a) # ["eva", "nikola"]
print(s)

a = ["ivan", 3, 4, 5, None, [{"eva": [3, ["nikola"]], 1: 'momo'}]]
s = strings(a) # ["ivan", "eva", "nikola"]
print(s)