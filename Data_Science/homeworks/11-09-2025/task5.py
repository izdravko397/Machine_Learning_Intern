import numpy as np

#  1 1 1 1
#  1 1 1 1
#  1 1 1 1
mat = np.ones((4, 4), dtype=np.int8)
# print(mat)
#=============================================

#  1 0 0 0
#  0 1 0 0
#  0 0 1 0
#  0 0 0 1

mat = np.eye(4, 4, dtype=np.int8, )

# print(mat)
#=============================================

#  1 0 0 0
#  0 2 0 0
#  0 0 3 0
#  0 0 0 4

mat = np.zeros((4, 4), dtype=np.int8)
np.fill_diagonal(mat, np.arange(1, 5))

# print(mat)
#=============================================

#   0  1  1 1
#  -1  0  1 1
#  -1 -1  0 1
#  -1 -1 -1 0 

mat = np.ones((4, 4), dtype=np.int8)

for i in range(4):
    for j in range(i + 1):
        if i == j:
            mat[i, j] = 0
            continue
        mat[i, j] = -1

# print(mat)
#=============================================

#  0 1 2 3
#  1 0 1 2
#  2 1 0 1
#  3 2 1 0

mat = np.zeros((4, 4), dtype=np.int8)

for i in range(4):
    for j in range(4):
        mat[i, j] = abs(i - j)

# print(mat)
#=============================================
#  0 1 2 3
#  4 5 6 7
#  8 9 a b
#  c d e f

arr = [hex(i)[2:] for i in range(16)]
mat = np.array(arr).reshape((4, 4))
# print(mat)
#=============================================
