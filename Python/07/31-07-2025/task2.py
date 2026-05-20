from matrix_calass import Matrix

m = Matrix(12, 5, 0, type='H') 
print(m)
print(type(m[0, 0]))
print()

m = Matrix(12, 5, 's', type='u') 
print(m)
print(type(m[0, 0]))
print()

m = Matrix(12, 5, 0, type='f') 
print(m)
print(type(m[0, 0]))
print()

m = Matrix(12, 5, 0, type=2) 
print(m)
print(type(m[0, 0]))
# матрица с размер 12 x 5, всички елменти са 0 и са от тип unsigned short