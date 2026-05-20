from time import perf_counter as clock
from random import randint
from numba import jit, njit, int64
import numpy as np

N = 10000
arr = np.random.randint(-1000, 1000, size=N, dtype="int64")
print(arr)

# @jit(int64(int64[:]))
@njit
def bruteForce(a):
  l = len(a)
  max = 0
  for i in range(l):
    sum = 0
    j = i
    while j < l:
      sum += a[j]
      if sum > max:
        max = sum
      j += 1
  return max, sum

def run():
  start = clock()
  r = bruteForce(arr)
  end = clock()
  print((end - start) * 1000, 'ms', 'N=', N, 'result=', r)

run()
run()