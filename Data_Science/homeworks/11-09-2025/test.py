from data import Matrix
import data
import numpy as np

# m = Matrix(data=[[1, 2, 3], [4, 5 ,6], [7, 8, 9]])
m = Matrix(data=[[1, 2, 3, 4, 5], [6, 7 ,8, 9 ,10], [11, 12, 13, 14, 15]])
# print(m)
# print()
# m1 = m[0:2, 3:4]
# m2 = m[:, 3:4]
# m3 = m[1:, 3:4]
# m4 = m[0:2, 3]

# print('m1 = ' , m1)
# print('m2 = ' , m2)
# print('m3 = ' , m3)
# print('m4 = ' , m4)
#============================

# print(type(m1))
# m1[0] = 55
# print(m1)
# print(m)

# # m[0:2, 3:4] = 3
# # m[:, 3:4] = 1
# # m[1:, 3:4] = 0
# # m[0:2, 3] = 4

# print(m)
#==========================
# a = data.arange(1 ,10)
# a = data.zeros((3, 4))
# print(a)

# b = data.zeros_like(m)
# print(b)

# a = data.full((2, 4), 10.3)
# print(a)
#========================
# m = Matrix(data=[-1, -3, 0, 11, 21])

# bm = m > 10
# print(bm)
# [False, False, False, True, True]
#========================
# m1 = Matrix(data=[-1, -3, 0, 1, 2, 5, -3])
# d = Matrix(data=[[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2], [-12, -4], [3, 4]])

# m2 = d[m1 > 0]
# print(m2)
# [[0, 0],
#  [1, 2], 
#  [-12, -4]]
#=======================
# m = Matrix(data=[1, 2, 3, 4, 5])
# a = data.std(m)
# print(a)

# a = Matrix(data=[[1, 2], [3, 4]])
# print(data.std(a))
# # 1.1180339887498949 

# print(data.std(a, axis=0))
# # array([1.,  1.])

# print(data.std(a, axis=1))
# # # array([0.5,  0.5])
#====================

# arr = data.zeros((8, 4))
# for i in range(8):
#   arr[i] = i
# print(arr)
# #[[0., 0., 0., 0.],
# # [1., 1., 1., 1.],
# # [2., 2., 2., 2.],
# # [3., 3., 3., 3.],
# # [4., 4., 4., 4.],
# # [5., 5., 5., 5.],
# # [6., 6., 6., 6.],
# # [7., 7., 7., 7.]])

# b = arr[[4, 3, 0, 6]]
# print(b)
# # [[4., 4., 4., 4.],
# #  [3., 3., 3., 3.],
# #  [0., 0., 0., 0.],
# #  [6., 6., 6., 6.]]

# arr = data.arange(32).reshape(8, 4)
# c = arr[[1, 5, 7, 2], [0, 3, 1, 2]]
# print(c)
# # [4, 23, 29, 10]


        # if isinstance(idx, slice): # !!!!!!!!!!!!!!!!!!!!!!
        #     start, stop, step = idx.start or 0, idx.stop or len(self), idx.step or 1
        #     new_start = self._start + start * self._step
        #     new_step = self._step * step
        #     new_stop = self._start + stop * self._step
        #     return RangeIndex(new_stop, new_start, new_step, self.name)