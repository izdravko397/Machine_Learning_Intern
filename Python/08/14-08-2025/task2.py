from data import Matrix
import data
import random

m = Matrix(2, 3, dtype=data.uint, fill=lambda x, y: 1) # запълва матрицата с 1ци
print(m)

m = Matrix(5, 5, dtype=data.uint, fill=lambda x, y: random.randint(0, 10)) 
print(m)

# print(m.submatrix(slice(0, 3), slice(0, 3)))
# запълва матрицата със случайни числа от 0 до 10