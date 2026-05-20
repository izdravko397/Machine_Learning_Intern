from data import Matrix
import data

m = Matrix(rows=4, cols=4, fill=lambda x, y: 1)
print(m)

#  1 1 1 1
#  1 1 1 1
#  1 1 1 1

m = Matrix(rows=4, cols=4, fill=lambda x, y: 1 if x == y else 0)
print(m)
 
#  1 0 0 0
#  0 1 0 0
#  0 0 1 0
#  0 0 0 1
 
m = Matrix(rows=4, cols=4, fill=lambda x, y: x + 1 if x == y else 0)
print(m)

#  1 0 0 0
#  0 2 0 0
#  0 0 3 0
#  0 0 0 4

m = Matrix(rows=4, cols=4, fill=lambda x, y: 0 if x == y else 1 if x < y else -1, dtype=data.int64)
print(m)
 
#   0  1  1 1
#  -1  0  1 1
#  -1 -1  0 1
#  -1 -1 -1 0 

m = Matrix(rows=4, cols=4, fill=lambda x, y: 0 if x == y else y - x if x < y else x - y)
print(m)
 
#  0 1 2 3
#  1 0 1 2
#  2 1 0 1
#  3 2 1 0

m = Matrix(rows=4, cols=4, fill=lambda x, y: (x * 4) + y)
print(m)
 
#  0 1 2 3
#  4 5 6 7
#  8 9 a b
#  c d e f