class Average:
    def __init__(self, n):
        self.n = n
        self.sequence = []
        self.inx = 0
        self.sum = 0

    def next(self, num):
        if len(self.sequence) == self.n:
            self.sum -= self.sequence[self.inx]
            self.sum += num
            self.sequence[self.inx] = num

            self.inx = 0 if self.inx + 1 == self.n else self.inx + 1
        else:
            self.sequence.append(num)
            self.sum += num
        
        return self.sum / len(self.sequence)
    
avg = Average(3)

val = avg.next(1) 
print(val) # 1
val = avg.next(2)
print(val) # 1.5
val = avg.next(3) 
print(val) # 2
val = avg.next(4) 
print(val) # 3
val = avg.next(5) 
print(val) # 4
val = avg.next(0) 
print(val) # 3