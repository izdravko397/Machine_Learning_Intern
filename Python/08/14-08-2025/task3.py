class SymetricMatrix:
    def __init__(self, size, default=0):
        self.size = size
        self.default = default
        self.matrix = {}

    def _check_r_c(self, row, col):
        if not (0 <= row <= self.size - 1) or not (0 <= col <= self.size - 1):
            raise IndexError('index out of range')
        
        return (row, col) if row <= col else (col, row)

    def __getitem__(self, key):
        row, col = self._check_r_c(*key)
        return self.matrix.get((row, col), self.default)
    
    def __setitem__(self, key, value):
        row, col = self._check_r_c(*key)
        self.matrix[(row, col)] = value

    def __str__(self):
        return '\n'.join(' '.join(str(self.matrix.get((r, c) if r <= c else (c, r), self.default))
                                  for c in range(self.size))
                                  for r in range(self.size))

sm = SymetricMatrix(5, 1)

# sm[1, 3] = 0
# print(sm[3, 1])
# # 0
# print(sm)


a = 1
c = 0

for i in range(5):
    for j in range(i, 5):
        print(a, end=' ')
        c += 1
        a += 1
    print('\n')
    a += (i + 1)

print(c)