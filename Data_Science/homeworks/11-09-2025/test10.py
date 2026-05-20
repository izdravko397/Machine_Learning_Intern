import data
from data import Matrix

x = [1, 2, 3]
y = [10, 20]

# x = [1, 2]
# y = [10, 20]

# x = [5, 6, 7]
# y = [100, 200]

# x = [1, 2, 3]
# y = [10, 20, 30, 40]

# coordinates = data.meshgrid(x, y)

# for c in coordinates:
#     print(c)
#     print()

# coordinates = data.meshgrid(x, y, indexing='ij')

# for c in coordinates:
#     print(c)
#     print()


m = Matrix(data=[[1, 2, 3], [5, 6, 7]])
v = Matrix(data=[[5, 6, 7], [1, 2, 3]])

# print(data.vstack(m, v))
# print(data.hstack(m, v))


m = Matrix(data=[[1, 2, 3], [4, 5, 6]])
m = Matrix(data=[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10,11,12]])

# for i in data.split(m, [1, 2, 3]):
#     print(i)
#     print()

m = Matrix(data=[[1, 2, 3, 4],
              [5, 6, 7, 8]])
m = Matrix(data=[1, 2, 3, 4, 5, 6])


# for i in data.split(m, 2):
#     print(i)
#     print()


m = Matrix(data=[[1, 2, 3, 4],
              [5, 6, 7, 8]])

m = Matrix(data=[1, 2, 3, 4, 5, 6])

a = Matrix(data=[[1, 2], [3, 4]])
print(data.tile(a, (3, 2)))
# print(data.tile(m, (2, 2)))
