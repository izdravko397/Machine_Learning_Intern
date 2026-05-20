from data import Matrix
import data
import numpy as np

# матрица с размер 12 x 5, всички елменти са 0 и са от тип unsigned short
# # m = np.full((3, 3), 0, data.ushort)
# print(m)
# print(Matrix.ARR_TCODE)

m = Matrix(12, 5, 3333, data.ushort) 
print(m)
print(type(m[0][0]))

m = Matrix(4, 3, 0, data.ushort)
print(m)
print(type(m[0][0]))

a = Matrix(2, 2, 3.14, data.float32)
print(a)
print(type(a[0][0]))

m = Matrix(3, 1, -7, data.int32)
print(m)
print(type(m[0][0]))

m = Matrix(5, 5, 1.2, np.float16)
print(m)
print(type(m[0][0]))

m = Matrix(5, 5, 'sss', data.uint)
print(m)
print(type(m[0][0]))