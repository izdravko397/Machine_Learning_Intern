from task1 import Matrix
from range import Range

m = Matrix(4, 4)
for i in range(4): 
  m.set(i, i, i + 1)
  
# // 1 0 0 0
# // 0 2 0 0
# // 0 0 3 0
# // 0 0 0 4

cols = Range(1, 2)
rows = Range(2, 3)
m2 = m.sub_matrix(cols, rows)

print(m2)
print(m2.cols)
print(m2.rows)

m2.set(1, 0, 3)
print(m2)