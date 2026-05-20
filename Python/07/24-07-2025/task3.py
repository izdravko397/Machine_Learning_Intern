class Range:
    def __init__(self, start, end, step=1):
        self.start = start
        self.end = end
        self.step = step

    def contains(self, n):
        i = self.start

        while i <= self.end:
            if n == i:
                return True           
            i += self.step

        return False
    
    def overlaps(self, other: 'Range'):
        return self.start <= other.end and self.end >= other.start
    
    def merge(self, other: 'Range'):
        if self.overlaps(other):
            self.start = other.start if other.start < self.start else self.start
            self.end = other.end if other.end > self.end else self.end
            return True
        
        return False
    
    def __repr__(self):
        return f'[{self.start}, {self.end}]'

r1 = Range(1, 3)
r2 = Range(2, 5)

print(r1.contains(2))
print(r1.overlaps(r2))

print(r1.merge(r2))
print(r1) # [1, 5]