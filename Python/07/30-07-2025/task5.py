from matrix_calass import Matrix

m = Matrix(2, 2, 3)
for i in range(m.rows):
    m[i, i] = i + 1

m1 = 2 + m # всички елмементи в резултатната матрица са 5
print(m1)
m1 = m + 2 # всички елмементи в резултатната матрица са 5
print(m1)

m2 = 3 * m # всички елмементи в резултатната матрица са 9
print(m2)
print(m * 3)

print(m - 1)
print(1 - m)

print(m / 2)
print(2 / m)