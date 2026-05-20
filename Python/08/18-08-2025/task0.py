from array import array

class SymetricMatrix:
    def __init__(self, size, default=0):
        self.size = size
        self.default = default
        count = size * (size + 1) // 2
        self.matrix = array('i', (default for _ in range(count)))

    def _check_r_c(self, row, col):
        if not (0 <= row < self.size) or not (0 <= col < self.size):
            raise IndexError('index out of range')
        return (row, col) if row <= col else (col, row)

    def _get_inx(self, row, col):
        return row * self.size - (row * (row - 1)) // 2 + (col - row)

    def __getitem__(self, key):
        row, col = self._check_r_c(*key)
        return self.matrix[self._get_inx(row, col)]
    
    def __setitem__(self, key, value):
        row, col = self._check_r_c(*key)
        self.matrix[self._get_inx(row, col)] = value

    def __str__(self):
        return '\n'.join(
            ' '.join(str(self.matrix[self._get_inx(*self._check_r_c(r, c))])
                    for c in range(self.size))
                    for r in range(self.size))

m = SymetricMatrix(4, default=0)
m[0, 1] = 5
m[2, 3] = 9
print(m)

sm = SymetricMatrix(5, 1)
print(sm)

sm[1, 3] = 0
print(sm[3, 1])
# 0
print(sm)