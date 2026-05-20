from matrix_calass import Matrix

m = Matrix(5, 5)

for i in range(m.rows):
    m[i, i] = i + 1

print(m)
print()

m[4, 1] = 3
print(m)
print()

m[:1] = [1, 1 ,1 , 1, 1]
print(m)
print()

m[:2, :3] = [1, 2, 3]
print(m)
print()

m[4, 2:4] = [1, 3]
print(m)
print()

m[1:4, 3] = 8
print(m)
print()

m[..., 3:] = [8, 9]
print(m)
print()

m[3:, 4] = 8
print(m)
print()


b = m[4, 1]
print(b)