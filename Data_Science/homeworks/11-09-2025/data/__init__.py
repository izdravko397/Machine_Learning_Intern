from .matrix import Matrix
import numpy as np
import array
import math

ushort = np.uint16
short = np.int16
uint = np.uint32

int8 = np.int8
uint8 = np.uint8
int16 = np.int16
uint16 = np.uint16
int32 = np.int32
uint32 = np.uint32
int64 = np.int64
uint64 = np.uint64
float64 = np.float64
float32 = np.float32

def arange(stop, start=None, step=None, dtype=int64):
    if start is not None:
        start, stop = stop, start
    else:
        start = 0

    step = 1 if not step else step

    if (start > stop and step > 0) or (start < stop and step < 0):
        raise IndexError('invalid params')

    data = [n for n in range(start, stop, step)]

    return Matrix(data=data, dtype=dtype)
        
def zeros(shape, dtype=int32):
    if isinstance(shape, int):
        return Matrix(rows=1, cols=shape, default_value=0, dtype=dtype)
    elif isinstance(shape, tuple):
        return Matrix(rows=shape[0], cols=shape[1], default_value=0, dtype=dtype)
    else:
        raise TypeError('invalid shape')


def zeros_like(array: Matrix, dtype=None):
    rows, cols = array.shape
    
    if not dtype:
        dtype = array.dtype

    return Matrix(rows=rows, cols=cols, default_value=0, dtype=dtype)


def full(shape, fill_value, dtype=None):
    if dtype is None:
        if isinstance(fill_value, float):
            dtype = float64
        elif isinstance(fill_value, int):
            dtype = int64

    if isinstance(shape, int):
        return Matrix(rows=1, cols=shape, default_value=fill_value, dtype=dtype)
    if isinstance(shape, tuple):
        return Matrix(rows=shape[0], cols=shape[1], default_value=fill_value, dtype=dtype)
    
    raise TypeError('invalid shape')
    
def std(a: Matrix, axis=None, dtype=float64):
    if axis is None:
        mean = sum(a.buffer) / len(a.buffer)
        diff_arr = a - mean
        sq_diff = diff_arr**2

        mean_sq_diff = sum(sq_diff.buffer) / len(sq_diff.buffer)
        return math.sqrt(mean_sq_diff)
    
    if axis == 0:
        a = a.T

    data = []

    for i in range(a.rows):
        row = []
        for j in range(a.cols):
            row.append(a[i, j])

        _sum = sum(row)
        mean = _sum / a.cols
        row = Matrix(dtype=a.dtype, data=row)
        diff_arr = row - mean
        sq_diff = diff_arr**2

        mean_sq_diff = sum(sq_diff.buffer) / len(sq_diff.buffer)
        data.append(math.sqrt(mean_sq_diff))
        
    return Matrix(dtype=dtype, data=data)
    
def in1d(arr1, arr2):
    if not isinstance(arr1, Matrix):
        arr1 = Matrix(data=arr1)

    if arr1.rows > 1:
        raise ValueError('requared 1d matrices')
    
    set_arr2 = set(arr2)

    if arr1.rows != 1:
        raise ValueError('The in1d function, with requires 1d arrays')
    
    return Matrix(data=[1 if arr1[c] in set_arr2 else 0 for c in range(arr1.cols)])

def unique(arr: Matrix, axis=None):
    if axis is None:
        if arr.rows > 1:
            arr = arr.reshape(1, arr.rows * arr.cols)

        return Matrix(data=set(arr), dtype=arr.dtype)

    if axis == 0:
        data = set()
        for r in range(arr.rows):
            data.add(tuple(arr[r]))

        return Matrix(data=list(data), dtype=arr.dtype)
    
    if axis == 1:
        data = set()
        for c in range(arr.cols):
            col = []
            for r in range(arr.rows):
                col.append(arr[r, c])

            data.add(tuple(col))
        data = list(data)
        unique_cols = list(zip(*data))
        
        return Matrix(data=list(unique_cols), dtype=arr.dtype)

def intersect1d(arr1: Matrix, arr2: Matrix):
    if arr1.rows > 1 or arr2.rows > 1:
        raise ValueError('requared 1d matrices')

    return Matrix(data=set(arr1) & set(arr2), dtype=arr1.dtype)

def union1d(arr1: Matrix, arr2: Matrix):
    if arr1.rows > 1 or arr2.rows > 1:
        raise ValueError('requared 1d matrices')

    return Matrix(data=set(arr1) | set(arr2), dtype=arr1.dtype)

builtin_sum = sum
def sum(arr: Matrix, axis=None):
    if axis is None:
        return builtin_sum(arr.buffer)
    
    if axis == 0:
        data = []
        for c in range(arr.cols):
            val = 0
            for r in range(arr.rows):
                val += arr[r, c]
            data.append(val)

    elif axis == 1:
        data = []
        for r in range(arr.rows):
            val = 0
            for c in range(arr.cols):
                val += arr[r, c]
            data.append(val)

    return Matrix(data=data)

def cumsum(arr: Matrix, axis=None):
    if axis is None:
        if arr.rows > 1:
            arr.reshape(1, arr.rows * arr.cols)

        val = 0
        def fill(x, y):
            nonlocal val
            val += arr[x, y]
            return val
        return Matrix(rows=1, cols=arr.cols, fill=fill)
    
    data = []
    if axis == 0:
        for c in range(arr.cols):
            val = 0
            for r in range(arr.rows):
                if c == 0:
                    data.append([])

                val += arr[r, c]
                data[r].append(val)
            val = 0

    elif axis == 1:
        for r in range(arr.rows):
            val = 0
            data.append([])
            for c in range(arr.cols):
                val += arr[r, c]
                data[r].append(val)
            val = 0

    return Matrix(data=data)

def mean(arr, axis=None):
    if axis is None:
        return sum(arr) / len(arr)
    
    data = []
    if axis == 0:
        for c in range(arr.cols):
            val = 0
            for r in range(arr.rows):
                val += arr[r, c]

            data.append(val / arr.rows)
            val = 0

    elif axis == 1:
        for r in range(arr.rows):
            val = 0
            for c in range(arr.cols):
                val += arr[r, c]

            data.append(val / arr.cols)
            val = 0

    return Matrix(data=data, dtype=float64)

def meshgrid(x, y=None, indexing='xy'):
    if not isinstance(x, Matrix):
        x = Matrix(data=x)

    if x.rows != 1:
        raise ValueError('meshgrid requires 1D arrays')

    if y is None:
        return (x, )
    
    if not isinstance(y, Matrix):
        y = Matrix(data=y)

    if y.rows != 1:
        raise ValueError('meshgrid requires 1D arrays')
    
    x_len = x.cols
    y_len = y.cols
    if indexing == 'xy':
        x = Matrix(buffer=x.buffer * y_len, rows=y_len, cols=x_len)
        y = Matrix(buffer=array.array(y.arr_type, (i for i in y.buffer for _ in range(x_len))), 
                   rows=y_len, cols=x_len)

    elif indexing == 'ij':
        x = Matrix(buffer=array.array(x.arr_type, (i for i in x.buffer for _ in range(y_len))), 
                   rows=x_len, cols=y_len)
        y = Matrix(buffer=y.buffer * x_len, rows=x_len, cols=y_len)
    else:
        raise ValueError('invalid indexing')
    
    return (x, y)

def vstack(arr1, arr2):
    if not isinstance(arr1, Matrix):
        arr1 = Matrix(data=arr1)

    if not isinstance(arr2, Matrix):
        arr2 = Matrix(data=arr2)

    if arr1.cols != arr2.cols:
        raise ValueError('Matrices must have same columns')
    
    return Matrix(buffer=arr1.buffer + arr2.buffer, rows=arr1.rows + arr2.rows, cols=arr1.cols)

def hstack(arr1, arr2):
    if not isinstance(arr1, Matrix):
        arr1 = Matrix(data=arr1)

    if not isinstance(arr2, Matrix):
        arr2 = Matrix(data=arr2)

    if arr1.rows != arr2.rows:
        raise ValueError('Matrices must have same rows')
    
    return Matrix(rows=arr1.rows, cols=arr1.cols + arr2.cols,
                  fill=lambda x, y: arr1[x, y] if y < arr1.cols else arr2[x, y - arr1.cols])


def split(arr: Matrix, parts_or_index, axis=0):
    if not isinstance(arr, Matrix):
        arr = Matrix(data=arr)

    if axis == 0:
        if isinstance(parts_or_index, int):
            if arr.rows > 1:
                if arr.rows % parts_or_index != 0:
                    raise ValueError('for axis 0 number of parts must be divisible to matrix rows')
                
                part_len = arr.rows // parts_or_index
            else:
                if arr.cols % parts_or_index != 0:
                    raise ValueError('for axis 0 number of parts must be divisible to matrix cols')
                
                part_len = arr.cols // parts_or_index

            return [arr[part_len * i:part_len * (i + 1)] for i in range(parts_or_index)]
        
        elif isinstance(parts_or_index, list):
            end = arr.rows if arr.rows > 1 else arr.cols
            res = []

            for i in range(len(parts_or_index)):
                if i == 0:
                    inx1 = 0
                    inx2 = parts_or_index[i]
                elif i == len(parts_or_index) - 1:
                    inx1 = parts_or_index[i - 1]
                    inx2 = parts_or_index[i]
                    res.append(arr[inx1: inx2])

                    inx1 = parts_or_index[i]
                    inx2 = end
                else:
                    inx1 = parts_or_index[i - 1]
                    inx2 = parts_or_index[i]

                res.append(arr[inx1: inx2])

            if len(parts_or_index) == 1:
                res.append(arr[parts_or_index[0]:])

            return res
        else:
            raise TypeError(f'Invalid 2 param {parts_or_index}')
        
    elif axis == 1:
        if isinstance(parts_or_index, int):
            if arr.cols % parts_or_index != 0:
                raise ValueError('for axis 1 number of parts must be divisible to matrix cols')
            
            part_len = arr.cols // parts_or_index
            return [arr[:, part_len * i:part_len * (i + 1)] for i in range(parts_or_index)]
        
        elif isinstance(parts_or_index, list):
            end = arr.cols
            res = []

            for i in range(len(parts_or_index)):
                if i == 0:
                    inx1 = 0
                    inx2 = parts_or_index[i]
                elif i == len(parts_or_index) - 1:
                    inx1 = parts_or_index[i - 1]
                    inx2 = parts_or_index[i]
                    res.append(arr[:, inx1: inx2])

                    inx1 = parts_or_index[i]
                    inx2 = end
                else:
                    inx1 = parts_or_index[i - 1]
                    inx2 = parts_or_index[i]

                res.append(arr[:, inx1: inx2])

            if len(parts_or_index) == 1:
                res.append(arr[:, parts_or_index[0]:])

            return res
        else:
            raise TypeError(f'Invalid 2 param {parts_or_index}')
    else:
        raise ValueError(f'axis can be 0 or 1 not : {axis}')
    

def vsplit(arr: Matrix, parts_or_index):
    if not isinstance(arr, Matrix):
        arr = Matrix(data=arr)

    if arr.rows == 1:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    
    return split(arr, parts_or_index)

def hsplit(arr: Matrix, parts_or_index):
    return split(arr, parts_or_index, axis=1)

def tile(arr, reps):
    if not isinstance(arr, Matrix):
        arr = Matrix(data=arr)

    if isinstance(reps, int):
        if reps >= 2:
            res = hstack(arr, arr)
            for _ in range(reps - 2):
                res = hstack(res, arr)
        elif reps == 1:
            res = arr
        else:
            res = Matrix(rows=0, cols=0, fill=lambda x, y: x)

        return res
    
    elif isinstance(reps, tuple):
        reps_len = len(reps)

        if not (1 <= reps_len <= 2):
            raise ValueError('tuple must be with 1 or 2 elements')

        if reps_len == 1:
            return tile(arr, reps[0])
        
        for_row = reps[0]
        for_col = reps[1]

        if for_col == 0 or for_row == 0:
            return Matrix(rows=0, cols=0, fill=lambda x, y: x)

        res = tile(arr, for_col)
        
        if for_row >= 2:
            res = vstack(res, res)
            for _ in range(for_row - 2):
                res = vstack(res, res)

        return res
    else:
        raise TypeError(f'reps must ne int or tuple not: {type(reps).__name__}')
        