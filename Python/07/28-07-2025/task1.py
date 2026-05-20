from range import Range

class Matrix:
    def __init__(self, cols, rows, default_value=0, submatrix=None):
        self.cols = cols
        self.rows = rows
        self.default_value = default_value
        self.matrix = [[self.default_value] * self.cols for _  in range(self.rows)] if submatrix == None else submatrix

    def set(self, x, y, value):
        try:
            self.matrix[x][y] = value
        except IndexError as e:
            print(e)

    def get(self, x, y):
        try:
            return self.matrix[x][y]
        except IndexError as e:
            print(e)

    def sub_matrix(self, columns: 'Range', rows: 'Range'):
        submatrix = [row[columns.start:columns.end + 1] for row in self.matrix[rows.start:rows.end + 1]]
        return Matrix(columns.end - columns.start + 1, rows.end - rows.start + 1, submatrix=submatrix)
    
    def __repr__(self):
        return '\n'.join(' '.join(str(num) for num in row) for row in self.matrix)


# mat = Matrix(3, 3)

# mat.set(0, 0, 3)
# print(mat.get(0, 0))
# print(mat.get(1, 1))
    
# m = Matrix(2, 2, -1)

# m.set(0, 1, 15)
# v1 = m.get(0, 1) # 15
# v2 = m.get(0, 0) # -1

# print(v1)
# print(v2)