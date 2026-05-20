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

    def __init__(self, cols=None, rows=None, dtype=np.int64, fill=None, default_value=None, data=None,
                buffer=None, offset=0, stride=None):
        if not (arr_type := Matrix.numpy_to_array_type.get(dtype.__name__)):
            raise TypeError(f'Unsupported dtype {dtype}')
        self.arr_type = arr_type
        self.dtype = dtype
        self.counter = 0
        
        self.offset = offset
        self.stride = stride if stride is not None else cols
        
        if buffer is not None:
            self.buffer = buffer
            self.rows = rows
            self.cols = cols
            return

        if data != None:
            if any(arg != None for arg in [cols, rows, fill, default_value]):
                raise TypeError("Cannot provide data together with cols, rows, fill or default_value")
            
            if not isinstance(data, set) and isinstance(data[0], (list, tuple)):
                if any(len(row) != len(data[0]) for row in data):
                    raise ValueError('Every row must have same columns')
                
                self.cols = len(data[0])
                self.rows = len(data)
                self.stride = self.cols

                self.buffer = array.array(arr_type, (data[i][j] for i in range(self.rows) for j in range(self.cols)))
            else:
                self.rows = 1
                self.cols = len(data)
                self.stride = self.cols
                self.buffer = array.array(arr_type, (item for item in data))
            return
    
        if cols == None or rows == None:
            raise TypeError('Rows and cols are required arguments')
        
        self.cols = cols
        self.rows = rows

        if fill != None and default_value != None:
            raise TypeError('Cannot provide both fill and default_value')

        if fill != None:
            self.buffer = array.array(arr_type, (fill(row, col) for row in range(rows) for col in range(cols)))

        elif default_value != None:
            self.buffer = array.array(arr_type, (default_value for _ in range(rows * cols)))

        else:
            raise TypeError('Must implement one of arguments: data, fill, default_value')
            
    @property
    def shape(self):
        return (self.rows, self.cols)
    
    @property
    def T(self):
        get_num = lambda r, c: self.buffer[c * self.cols + r]
        return Matrix(rows=self.cols, cols=self.rows, fill=get_num)
                
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key

            if isinstance(row, int) and isinstance(col, int):
                self.buffer[self.offset + row * self.stride + col] = value

            else:
                if not isinstance(row, slice):
                    row = slice(row, row + 1)
                if not isinstance(col, slice):
                    col = slice(col, col + 1)

                r_start, r_stop = row.start or 0, row.stop or self.rows
                c_start, c_stop = col.start or 0, col.stop or self.cols

                for r in range(r_start, r_stop):
                    for c in range(c_start, c_stop):
                        self.buffer[self.offset + r*self.stride + c] = value

        elif isinstance(key, slice):
            if self.rows == 1:
                start, stop = key.start or 0, key.stop or self.cols
                for c in range(start, stop):
                    self.buffer[self.offset + c] = value
            else:
                start, stop = key.start or 0, key.stop or self.rows
                for r in range(start, stop):
                    for c in range(self.cols):
                        self.buffer[self.offset + r*self.stride + c] = value

        elif isinstance(key, int):
            if self.rows == 1:
                self.buffer[self.offset + key] = value
            else:
                for c in range(self.cols):
                    self.buffer[self.offset + key*self.stride + c] = value

    def __getitem__(self, key):
        if isinstance(key, Matrix):
            data = []
            for i in range(key.cols):
                if key[i]:
                    if self.rows == 1:
                        data.append(self[i])
                    else:
                        data.append([self[i, j] for j in range(self.cols)])

            return Matrix(data=data, dtype=self.dtype)
        
        if isinstance(key, list):
            if self.rows > 1:
                data = []
                for i in key:
                    data.append([self[i, j] for j in range(self.cols)])

                return Matrix(dtype=self.dtype, data=data)

            return Matrix(dtype=self.dtype, data=[self[inx] for inx in key])

        if isinstance(key, tuple):
            row, col = key

            if isinstance(row, list) and isinstance(col, list):
                return Matrix(dtype=self.dtype, data=[self[r, c] for r, c in zip(row, col)])

            if isinstance(row, int) and isinstance(col, int):
                return self.buffer[self.offset + row * self.stride + col]
            
            if not isinstance(row, slice):
                row = slice(row, row + 1)
            if not isinstance(col, slice):
                col = slice(col, col + 1)

            r_start, r_stop = row.start or 0, row.stop or self.rows
            c_start, c_stop = col.start or 0, col.stop or self.cols

            r_step = row.step or 1 
            c_step = col.step or 1

            if r_step != 1 or c_step != 1:
                data = []
                for r in range(r_start, r_stop, r_step):
                    row = []
                    for c in range(c_start, c_stop, c_step):
                        row.append(self[r, c])
                    data.append(row)

                return Matrix(data=data, dtype=self.dtype)
            
            new_rows = r_stop - r_start
            new_cols = c_stop - c_start
            new_offset = self.offset + r_start * self.stride + c_start

            return Matrix(rows=new_rows, cols=new_cols, offset=new_offset, stride=self.stride,
                           dtype=self.dtype, buffer=self.buffer)
        
        if isinstance(key, slice):
            if self.rows == 1:
                c_start, c_stop = key.start or 0, key.stop or self.cols
                c_step = key.step or 1
                if c_step != 1:
                    indexes = list(range(c_start, c_stop, c_step))
                    return self[indexes]
                
                new_cols = c_stop - c_start
                new_offset = self.offset + c_start
                return Matrix(rows=self.rows, cols=new_cols, offset=new_offset, stride=self.stride,
                              dtype=self.dtype, buffer=self.buffer)
            
            r_start, r_stop = key.start or 0, key.stop or self.rows
            r_step = key.step or 1
            if r_step != 1:
                    indexes = list(range(r_start, r_stop, r_step))
                    return self[indexes]
            
            new_rows = r_stop - r_start
            new_offset = self.offset + r_start * self.stride
            return Matrix(rows=new_rows, cols=self.cols, offset=new_offset, stride=self.stride,
                              dtype=self.dtype, buffer=self.buffer)
        
        if isinstance(key, int):
            if self.rows == 1:
                return self.buffer[self.offset + key]
                
            new_offset = self.offset + key * self.stride
            return Matrix(rows=1, cols=self.cols, offset=new_offset, stride=self.stride,
                              dtype=self.dtype, buffer=self.buffer)
        
    def reshape(self, row, col):
        if row * col != self.rows * self.cols:
            raise ValueError('incompatible capacity')
            
        self.rows = row
        self.cols = col
        self.stride = col

        return self
    
    def dot(self, other):
        if self.cols != other.rows:
            raise ValueError(f'To multiply matrices other matrix must have {self.cols} rows')
        
        data = []
        for self_r in range(self.rows):
            row = []
            for other_c in range(other.cols):
                val = 0
                for other_r in range(other.rows):
                    val += self[self_r, other_r] * other[other_r, other_c]
                row.append(val)
            data.append(row)

        return Matrix(data=data, dtype=np.float64)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.rows * self.cols == self.counter:
            self.counter = 0
            raise StopIteration
        
        val = self[self.counter]
        self.counter += 1
        return val
    
    def __len__(self):
        return len(self.buffer)
    
    def __gt__(self, other):
        return Matrix(rows=self.rows, cols=self.cols, dtype=np.int8, 
                      fill=lambda x, y: 1 if self[x, y] > other else 0)
    
    def __lt__(self, other):
        return Matrix(rows=self.rows, cols=self.cols, dtype=np.int8, 
                      fill=lambda x, y: 1 if self[x, y] < other else 0)
    
    def __eq__(self, other):
        return Matrix(rows=self.rows, cols=self.cols, dtype=np.int8, 
                      fill=lambda x, y: 1 if self[x, y] == other else 0)
    
    def __pow__(self, exponent):
        return Matrix(rows=self.rows, cols=self.cols, dtype=self.dtype, 
                      fill=lambda x, y: self[x, y] ** exponent)
        
    def submatrix(self, columns, rows):
        cols_slice = columns
        rows_slice = rows

        if not isinstance(columns, slice):
            col_start = columns[0]
            col_stop = columns[1]
            cols_slice = slice(col_start, col_stop)
        
        if not isinstance(rows, slice):
            row_start = rows[0]
            row_stop = rows[1]
            rows_slice = slice(row_start, row_stop)

        c_start, c_stop = cols_slice.start, cols_slice.stop
        r_start, r_stop = rows_slice.start, rows_slice.stop
        
        sub_matrix = [self.matrix[r * self.cols + c] for r in range(r_start, r_stop) for c in range(c_start, c_stop)]
        return Matrix(c_stop - c_start, r_stop - r_start, fill=lambda x, y: sub_matrix[x * (c_stop - c_start) + y])
    
    def _check_mat_shape(self, other):
        if self.rows != other.rows or self.cols != other.cols:
                raise ValueError('matrices must have same shape')

    def __add__(self, n):
        if isinstance(n, Matrix):
            self._check_mat_shape(n)
            
            return Matrix(rows=self.rows, cols=self.cols, 
                          fill=lambda x, y: self.buffer[x * self.cols + y] + n.buffer[x * self.cols + y])
        
        return Matrix(cols=self.cols, rows=self.rows, 
                      fill=lambda x, y: self.buffer[x * self.cols + y] + n)
    
    def __radd__(self, n):
        return self + n
    
    def __sub__(self, n):
        if isinstance(n, Matrix):
            self._check_mat_shape(n)
            
            return Matrix(rows=self.rows, cols=self.cols, dtype=np.float64, 
                          fill=lambda x, y: self.buffer[x * self.cols + y] - n.buffer[x * self.cols + y])
        
        return Matrix(dtype=np.float64, cols=self.cols, rows=self.rows, 
                      fill=lambda x, y: self.buffer[x * self.cols + y] - n)
    
    def __rsub__(self, n):
        if isinstance(n, Matrix):
            self._check_mat_shape(n)
            
            return Matrix(rows=self.rows, cols=self.cols, dtype=np.int64, 
                          fill=lambda x, y: n.buffer[x * self.cols + y] - self.buffer[x * self.cols + y])
        
        return Matrix(dtype=np.int64, cols=self.cols, rows=self.rows, 
                      fill=lambda x, y: n - self.buffer[x * self.cols + y])
    
    def __mul__(self, n):
        if isinstance(n, Matrix):
            self._check_mat_shape(n)
            
            return Matrix(rows=self.rows, cols=self.cols, dtype=np.int64, 
                          fill=lambda x, y: n.buffer[x * self.cols + y] * self.buffer[x * self.cols + y])

        return Matrix(cols=self.cols, rows=self.rows, dtype=np.int64, 
                      fill=lambda x, y: self.buffer[x * self.cols + y] * n)
    
    def __rmul__(self, n):
        return self * n

    def __truediv__(self, n):
        if isinstance(n, Matrix):
            self._check_mat_shape(n)
            
            return Matrix(rows=self.rows, cols=self.cols, dtype=np.float64, 
                          fill=lambda x, y: self.buffer[x * self.cols + y] / n.buffer[x * self.cols + y])
        
        return Matrix(dtype=np.float64, cols=self.cols, rows=self.rows, 
                      fill=lambda x, y: self.buffer[x * self.cols + y] / n)
    
    def __rtruediv__(self, n):
        if isinstance(n, Matrix):
            self._check_mat_shape(n)
            
            return Matrix(rows=self.rows, cols=self.cols, dtype=np.int64, 
                          fill=lambda x, y: n.buffer[x * self.cols + y] / self.buffer[x * self.cols + y])
        
        return Matrix(dtype=np.float64, cols=self.cols, rows=self.rows, 
                      fill=lambda x, y: n / self.buffer[x * self.cols + y])
    
    # def __str__(self):
    #     return '[' + '\n'.join(
    #         '[' + ', '.join(str(self.matrix[r * self.cols + c]) for c in range(self.cols)) + ']'
    #         for r in range(self.rows)) + ']'

    # def __str__(self):
    #     max_len = max(len(str(el)) for el in self.matrix)
    #     return '\n'.join(
    #         ' '.join(f'{self.matrix[r * self.cols + c]:>{max_len}}' for c in range(self.cols))
    #         for r in range(self.rows))

    # def __str__(self):
    #     max_each_col_len = []

    #     for c in range(self.cols):
    #         col_len = set()
    #         for r in range(self.rows):
    #             col_len.add(len(str(self.matrix[r * self.cols + c])))
    #         max_each_col_len.append(max(col_len))
        
    #     return '\n'.join(
    #         ' '.join(f'{self.matrix[r * self.cols + c]:>{max_each_col_len[c]}}' for c in range(self.cols))
    #         for r in range(self.rows))

    # def __str__(self):
    #     max_len = max(len(str(el)) for el in self.matrix)
    #     return '\n'.join(
    #         ' '.join(f'{hex(self.matrix[r * self.cols + c]):>{max_len}}'[2:] for c in range(self.cols))
    #         for r in range(self.rows))

    def __str__(self):
        rows = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                # print(r, c)
                row.append(self.buffer[self.offset + r*self.stride + c])

            rows.append(str(row))
        return "\n".join(rows)