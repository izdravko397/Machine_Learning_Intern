class Range:
    def __init__(self, start, end, step=1):
        self.start = start
        self.pos = start
        self.end = end
        self.step = step

    def contains(self, n):
        for i in self:
            if i == n:
                return True
        return False
    
    def overlaps(self, other: 'Range'):
        return self.start <= other.end and self.end >= other.start
    
    def merge(self, other: 'Range'):
        if self.overlaps(other):
            self.start = other.start if other.start < self.start else self.start
            self.end = other.end if other.end > self.end else self.end
            return True
        
        return False
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.step > 0:
            condition = self.pos < self.end
        else:
            condition = self.pos > self.end

        if condition:
            res = self.pos
            self.pos += self.step
            return res
        else:
            raise StopIteration
        
    def __getitem__(self, key):
        return self.start if key == 0 else self.end
    
    def __repr__(self):
        return f'[{self.start}, {self.end}]'

# r1 = Range(1, 3)
# print()
# r2 = Range(2, 10, 3)
# print(r2.contains(5))

# print(r1.contains(2))
# print(r1.overlaps(r2))

# print(r1.merge(r2))
# print(r1) # [1, 5]


# for i in Range(5, 1, -1):
#     print(i)