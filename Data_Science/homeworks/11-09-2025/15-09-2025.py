from data import Matrix
import data

# m1 = Matrix(data=[[2, 2, 2], [3, 3, 3]])
# m2 = Matrix(data=[[2, 2], [3, 3], [4, 4]])

# res = m1.dot(m2)

# print(res)
#================
m = Matrix(data=[1, 2, 3, 4 ,10, 10, 3, 1])
a = [2, 10]
a = Matrix(data=a)

# mask = data.in1d(m, a)
# print(mask)

m1 = Matrix(data=[[1, 2, 3, 4 ],[10, 10, 3, 1]])

# unique = data.unique(m1)
# print(unique)

intersect = data.intersect1d(m, m)
print(intersect)


# x = Matrix(data=[1, 2, 2, 3])
# y = Matrix(data=[2, 3, 4])

# print(data.union1d(x, y))
#====================
# a = Matrix(data=[[1, 2, 3],
#               [4, 5, 6]])

# print(data.sum(a))      
# print(data.sum(a, axis=0))   
# print(data.sum(a, axis=1)) 

# a = Matrix(data=[1, 2, 3, 4])
# print(data.cumsum(a))
# # [ 1  3  6 10 ]

b = Matrix(data=[[1, 2, 3],
              [4, 5, 6]])

# print(data.cumsum(b))
# # [ 1  3  6 10 15 21 ]

# print(data.cumsum(b, axis=0))
# # [[ 1  2  3]
# #  [ 5  7  9]]

# print(data.cumsum(b, axis=1))
# # [[ 1  3  6]
# #  [ 4  9 15]]   (по редове)

a = Matrix(data=[[1, 2, 3],
              [4, 5, 6]])

# print(data.mean(a))# 3.5
# print(data.mean(a, axis=0))# [2.5 3.5 4.5] 
# print(data.mean(a, axis=1))# [2.0 5.0]
#=====================
# a = data.arange(11)
# b = a[::2] # 0, 2, 4 ..
# print(b)

# c = data.arange(100).reshape(10, 10)
# d = c[::2, ::2]
# print(d)
# [[0, 2, 4, 6, 8],
#  [10, 12, ... 18],
#  ..
#  [90, 92, ..., 98]]

b = Matrix(data=[[1, 2],
              [1, 2],
              [3, 4]])

# row1 = b[:1]
# print(row1)

# row2 = b[1:2]
# print(row2)
# print(row1 == row2)

# print(data.unique(b, axis=0))

c = Matrix(data=[[1, 2, 2, 3],
            [4, 5, 5, 6],
            [7, 8, 8, 9]])

res = data.unique(c, axis=1)
print(res)