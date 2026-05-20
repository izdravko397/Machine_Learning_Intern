from data import Matrix
import data

m = Matrix(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(m[3]) # 3
print(m[2:4])# [2, 3]


m[2:4] = 0
# [0, 1, 0, 0, 4, 5, 6, 7, 8, 9]

print(m)

mat = Matrix(data=[5, 1, 2])
sum = mat / 3

# print(sum)
# [0, 2, 4]

m = Matrix(data=[0, 1, 2, 4])
print(m)

ms = m.reshape(2, 2)
# [[0, 1],
#  [2, 3]]
print(m)
print(ms)