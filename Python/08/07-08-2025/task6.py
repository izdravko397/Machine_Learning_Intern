import time as t
from functools import wraps

def time(func):
    @wraps(func)
    def call(*args):
        start = t.time()
        result = func(*args)
        end = t.time()

        print(f'{func.__name__}{args}, time = {int((end - start) * 1000)} ms')
        return result
    return call

@time
def sort_test(arr, repeat):
  for _ in range(repeat):
    b = sorted(arr)
    
sort_test([3, 5, 5, 8, 12], 10000)
# sort_test([3, 5, 5, 8, 12), 10000), time=125 ms