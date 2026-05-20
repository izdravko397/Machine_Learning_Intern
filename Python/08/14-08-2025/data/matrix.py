import array
import numpy as np 

class Matrix:
    numpy_to_array_type = {
    'int8': 'b',
    'uint8': 'B',
    'int16': 'h',
    'uint16': 'H',
    'int32': 'i',
    'uint32': 'I',
    'int64': 'q',
    'uint64': 'Q',
    'float32': 'f',
    'float64': 'd',    
    }

    def __init__(self, cols, rows, dtype=np.ushort, fill=lambda x, y: int(str(x) + str(y))):
        self.cols = cols
        self.rows = rows

        if not (arr_type := Matrix.numpy_to_array_type.get(dtype.__name__)):
            raise TypeError
        
        self.matrix = array.array(arr_type, (fill(row, col) for row in range(rows) for col in range(cols)))

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key1, key2 = key

            if isinstance(key1, int) and isinstance(key2, int):
                self.matrix[key1 * self.cols + key2] = value

            else:
                if key1 is Ellipsis:
                    key1 = slice(0, self.rows)
                if key2 is Ellipsis:
                    key2 = slice(0, self.rows)

                for i in range(key1.start if key1.start != None else 0, key1.stop if key1.stop != None else self.rows):
                    self.matrix[i][key2] = value

        else:
            self.matrix[key] = [value]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key1, key2 = key

            if isinstance(key1, int) and isinstance(key2, int):
                return self.matrix[key1 * self.cols + key2]
            
            if key1 is Ellipsis:
                key1 = slice(0, self.rows)
            if key2 is Ellipsis:
                key2 = slice(0, self.rows)

            return [row[key2] for row in self.matrix[key1]]
        
        return self.matrix[key * self.cols:key * self.cols + self.cols]
        
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

        c_start, c_stop = cols_slice.start, cols_slice.stop
        r_start, r_stop = rows_slice.start, rows_slice.stop
        
        sub_matrix = [self.matrix[r * self.cols + c] for r in range(r_start, r_stop) for c in range(c_start, c_stop)]
        return Matrix(c_stop - c_start, r_stop - r_start, fill=lambda x, y: sub_matrix[x * (c_stop - c_start) + y])
    
    def __add__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[i + n for i in self.matrix])
    
    def __radd__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[i + n for i in self.matrix])
    
    def __sub__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[i - n for i in self.matrix])
    
    def __rsub__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[n - i for i in self.matrix])
    
    def __mul__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[i * n for i in self.matrix])
    
    def __rmul__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[i * n for i in self.matrix])

    def __truediv__(self, n):
        if n == 0:
            raise ZeroDivisionError
        return Matrix(self.cols, self.rows, submatrix=[i / n for i in self.matrix])
    
    def __rtruediv__(self, n):
        return Matrix(self.cols, self.rows, submatrix=[n / i or 1 for i in self.matrix])

    def __repr__(self):
        return '\n'.join(' '.join(str(self.matrix[r * self.cols + c]) for c in range(self.cols)) for r in range(self.rows))