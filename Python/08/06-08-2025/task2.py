class Average:
    def __init__(self, n):
        self.n = n
        self.sequence = []

    def next(self, num):
        self.sequence.append(num)

        if len(self.sequence) < self.n:
            res = sum(self.sequence) / len(self.sequence)
            return int(res) if int(res) == res else res
        
        sum_n_elements = 0
        for i in range(1, self.n + 1):
            sum_n_elements += self.sequence[-i]

        res = sum_n_elements / self.n
        return int(res) if int(res) == res else res
    
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