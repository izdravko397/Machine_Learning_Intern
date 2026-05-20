from matrix_calass import Matrix

m = Matrix(5, 5)
for i in range(m.rows):
    m[i, i] = i + 1
print(m)
print()

m1 = m.submatrix([1, 3], [2, 4])
print(m1)
print()

m1 = m.submatrix((1, 3), (2, 4))
print(m1)
print()

m1 = m.submatrix(slice(1, 3), slice(2, 4))
print(m1)
print()

m1 = m.submatrix(range(1, 3), range(2, 4))
print(m1)
print()

m1 = m.submatrix(slice(1, 3), [2, 4])
print(m1)
print()

m1 = m.submatrix((1, 3), range(2, 4))
print(m1)
print(m1.cols)

