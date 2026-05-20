import numpy as np
import array

class Matrix:
    # ARR_TYPECODES_NUMPY = {
    # 'b': np.int8,
    # 'B': np.uint8,
    # 'h': np.int16,
    # 'H': np.uint16,
    # 'i': np.int32,
    # 'I': np.uint32,
    # 'l': np.int64,
    # 'L': np.uint64,
    # 'q': np.int64,
    # 'Q': np.uint64,
    # 'f': np.float32,
    # 'd': np.float64
    # }

    ARR_TCODE = array.typecodes

    def __init__(self, cols, rows, default_value=0, dtype=np.ushort, submatrix=None):
        self.cols = cols
        self.rows = rows
        self.default_value = default_value

        # if dtype not in Matrix.ARR_TYPECODES_NUMPY.values():
        #     raise TypeError

        if np.dtype(dtype).char not in Matrix.ARR_TCODE:
            raise TypeError
        
        self.matrix = np.full((rows, cols), default_value, dtype=dtype) if submatrix == None else submatrix

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key1, key2 = key

            if isinstance(key1, int):
                self.matrix[key1][key2] = value

            else:
                if key1 is Ellipsis:
                    key1 = slice(0, self.rows)
                if key2 is Ellipsis:
                    key1 = slice(0, self.rows)

                for i in range(key1.start if key1.start != None else 0, key1.stop if key1.stop != None else self.rows):
                    self.matrix[i][key2] = value

        else:
            self.matrix[key] = [value]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key1, key2 = key

            if isinstance(key1, int):
                return self.matrix[key1][key2]
            
            if key1 is Ellipsis:
                key1 = slice(0, self.rows)
            if key2 is Ellipsis:
                key1 = slice(0, self.rows)

            return [row[key2] for row in self.matrix[key1]]
        
        return self.matrix[key]
        
    def submatrix(self, columns, rows):
        cols_slice = columns
        rows_slice = rows

        if not isinstance(columns, slice):
            col_start = columns[0]
            col_stop = columns[1]
            cols_slice = slice(col_start, col_stop + 1)
        
        if not isinstance(rows, slice):
            row_start = rows[0]
            row_stop = rows[1]
            rows_slice = slice(row_start, row_stop + 1)
        
        sub_matrix = [row[cols_slice] for row in self.matrix[rows_slice]]
        return Matrix(cols_slice.stop - cols_slice.start, rows_slice.stop - rows_slice.start, submatrix=sub_matrix)
    
    def __add__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[[el + n for el in row] for row in self.matrix])
    
    def __radd__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[[el + n for el in row] for row in self.matrix])
    
    def __sub__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[[el - n for el in row] for row in self.matrix])
    
    def __rsub__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[[n - el for el in row] for row in self.matrix])
    
    def __mul__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[[el * n for el in row] for row in self.matrix])
    
    def __rmul__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[[el * n for el in row] for row in self.matrix])

    def __truediv__(self, n):
        if n == 0:
            raise ZeroDivisionError
        return Matrix(self.cols, self.rows, submatrix=[[el / n for el in row] for row in self.matrix])
    
    def __rtruediv__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[[n / el if el != 0 else None for el in row] for row in self.matrix])

    def __repr__(self):
        return '\n'.join(' '.join(str(num) for num in row) for row in self.matrix)