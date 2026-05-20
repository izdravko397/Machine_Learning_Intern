# да се създаде nympy матрица с оценките на ученици, в която редовете отговарят на един 
# ученик а колоните на оценките получени през срока. матрицата да се запълни със случайни числа в 
# диапазона 30-100 (американска система за оценяване).

# за да са по справедливи оценките да се коригират така че средно-артиметичното на всички оценки 
# да бъде предварително зададаена стойност, примерно 70.

# да се изчисли оценката за семестъра за всеки ученик като средно аритметично от коригираните оценки.

# да не се ползват цикли, а само функции на numpy.


import numpy as np

students = 10
subjects = 10

low, high = 30, 100
target_average = 70

mat = np.random.randint(low, high + 1, (students, subjects))
print(mat)

average = mat.mean()
print('current mean: ', average)

normalized_mat = mat - average + target_average
print(normalized_mat)

mask_under_low = normalized_mat < low
normalized_mat[mask_under_low] = low

mask_greater_high = normalized_mat > high
normalized_mat[mask_greater_high] = high

normalized_mat = normalized_mat.astype(np.int64)

print('normalized mat:\n', normalized_mat)
print('normalized mat mean: ', normalized_mat.mean())

students_semester_grade = np.sum(normalized_mat, axis=1)
students_semester_grade = students_semester_grade / subjects
print(students_semester_grade)
