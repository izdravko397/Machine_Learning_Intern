class Average:
    def __init__(self, n):
        self.n = n
        self.sequence = []
        self.inx = 0
        self.sum = 0

    def next(self):
        num = yield

        while True:
            if len(self.sequence) == self.n:
                self.sum -= self.sequence[self.inx]
                self.sum += num
                self.sequence[self.inx] = num

                self.inx = 0 if self.inx + 1 == self.n else self.inx + 1
            else:
                self.sequence.append(num)
                self.sum += num
        
            num = yield self.sum / len(self.sequence)
    
avg = Average(3).next()
avg.send(None)

val = avg.send(1) 
print(val) # 1

val = avg.send(2)
print(val) # 1.5
# next(avg)

val = avg.send(3) 
print(val) # 2
# next(avg)

val = avg.send(4) 
print(val) # 3
# next(avg)

val = avg.send(5) 
print(val) # 4
# next(avg)

val = avg.send(0) 
print(val) # 3
# next(avg)

avg.close()