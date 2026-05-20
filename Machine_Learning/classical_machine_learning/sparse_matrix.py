import numpy as np

class SparseMatrix:
    def __init__(self, arr):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        self.dtype = arr.dtype
        self.shape = arr.shape if len(arr.shape) == 2 else (1, arr.shape[0])
        mask = arr != 0
        self.coords = np.where(mask)
        if self.shape[0] == 1:
            self.coords = (np.ones_like(self.coords[0]), self.coords[0])
        self.values = arr[mask]

    @staticmethod
    def _fill_vals(arr, row_inxs, col_inxs, vals):
        for (i, j), val in zip(zip(row_inxs, col_inxs), vals):
            arr[i, j] = val
    
    def toarray(self):
        arr = np.zeros(self.shape)
        self._fill_vals(arr, self.coords[0], self.coords[1], self.values)
        return arr

    def __repr__(self):
        head = f"<Compressed Sparse Row sparse matrix of dtype '{self.dtype}'\n\
        with {len(self.values)} stored elements and shape {self.shape}>\n"
        
        sep = '     '
        rows = [f'Coords{sep}Values']
        for i, ((i, j), val) in enumerate(zip(zip(*self.coords), self.values)):
            if i == 19:
                break
            rows.append(f'({i}, {j}){sep}{val}')

        return head + '\n'.join(rows)

# data = [
#     [0, 0, 3],
#     [4, 0, 0],
#     [0, 0, 5]
# ]
# mat = SparseMatrix(data)
# print(mat)
# print(repr(mat))
# print(mat[0])
# # print(mat[[1, 2]])

# from scipy.sparse import csr_matrix

# arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])
# mat = csr_matrix(arr)
# print(mat[0])
# # print(arr.shape)
# # print(mat[1, 5])
# # print(arr[0, 5])