def is_iter_obj(obj):
    # try:
    #     iter(obj)
    #     return True
    # except:
    #     return False
    
    return hasattr(obj, '__iter__')

class Enumerate:
    def __init__(self, iterable, start):
        self.iterable = iterable
        self.start = start
        self.iterable_inx = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iterable_inx == len(self.iterable):
            raise StopIteration
        
        inx_val = self.start
        self.start += 1

        iter_val = self.iterable[self.iterable_inx]
        self.iterable_inx += 1

        return (inx_val, iter_val)
    
def enumerate(iterable, start=0):
    if is_iter_obj(iterable) and isinstance(start, int):
        return Enumerate(iterable, start)
    raise TypeError

# test = [1, 2, 3, 4, 5, 6]
# for i, val in enumerate(test):
#     print(i, val)

# for val in enumerate(test):
#     print(val)

class Zip:
    def __init__(self, iterables):
        self.iterables = iterables
        self.index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        val = []

        for i in self.iterables:
            try:
                val.append(i[self.index])
            except:
                raise StopIteration
        
        self.index += 1
        return tuple(val)

def zip(*iterables):
    for i in iterables:
        if not is_iter_obj(i):
            raise TypeError
        
    return Zip(iterables)

# test = [1, 2, 3, 4, 5, 6]
# testa = [1, 2, 3, 4]
# testb = [1, 2, 3, 4, 5, 6]

# for val in zip(test, testb, testa):
#     print(val)

def all(iterator):
    if not is_iter_obj(iterator):
        raise TypeError
    
    for el in iterator:
        if not el:
            return False
    return True

# print(all([1, 2, 3]))
# print(all([1, 0, 3]))
# print(all([True, False, True]))
# print(all([True, True]))
# print(all([]))  

def any(iterator):
    if not is_iter_obj(iterator):
        raise TypeError
    
    for el in iterator:
        if el:
            return True
    return False

print(any([1, 2, 3]))            
print(any([0, 0, 5]))            
print(any([0, False, None, ""])) 
print(any([]))                   