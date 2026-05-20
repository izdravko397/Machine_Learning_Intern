class Fibonacci:
    def __init__(self, n):
        self.n = n
        self.counter = n
        self.previous_num = 1
        self.before_previous_num = 0

    def __iter__(self):
        return self
   
    def __next__(self):
        if self.counter == 0:
            raise StopIteration
        
        if self.counter > self.n - 1:
            self.counter -= 1
            return self.before_previous_num
        elif self.counter > self.n - 2:
            self.counter -= 1
            return self.previous_num
        
        self.counter -= 1
        val = self.before_previous_num + self.previous_num
        self.before_previous_num, self.previous_num = self.previous_num, val
        return val
        
def fib(n):
    return Fibonacci(n)

for n in fib(10):
  print(n)   # 1 1 3 4 .....