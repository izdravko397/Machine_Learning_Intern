class Counter:
    def __init__(self, iterable):
        self.dict = self.filling_the_dictionary(iterable)
        
    def filling_the_dictionary(self, iter):
        res = {}
        for i in iter:
            if i not in res:
                res[i] = 0
            res[i] += 1
        
        return res
    
    def __getitem__(self, name):
        if val := self.dict.get(name):
            return val
        else:
            raise AttributeError
        
    # def __setitem__(self, name, value):
    #     self.dict[name] = value
    
    def __str__(self):
        return f'Counter({dict(sorted(self.dict.items(), key=lambda items: items[1], reverse=True))})'

list = ['x','y','z','x','x','x','y', 'z']
c = Counter(list)
print(c) # Counter({'x': 4, 'y': 2, 'z': 2})
print(c['x']) # 4

str = "Hello from Infinno"
c = Counter(str)
print(c)
# Counter({'o': 3, 'n': 3, 'l': 2, ' ': 2, 'f': 2, 'H': 1, 'e': 1, 'r': 1, 'm': 1, 'I': 1, 'i': 1})