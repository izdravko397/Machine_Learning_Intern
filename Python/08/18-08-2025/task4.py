"""
да се създаде property Т към класа Matrix, което да връща транспонирана матрица
Пример:

m = Matrix(data=[[1, 2, 3], [4, 5, 6]])
print(m)
print(m.T)
Output:

[[1, 2, 3],
 [4, 5, 6]]

[[1, 4],
 [2, 5],
 [3, 6]]
"""

from data import Matrix

m = Matrix(data=[[1, 2, 3], [4, 5, 6]])
print(m)
# print(m.T)

# m = Matrix(rows=2, cols=3, fill=lambda x, y: x + y)
# print(m)
# print(m.T)