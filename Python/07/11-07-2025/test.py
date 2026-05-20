# s = 'gay'
# print(s[::-1])

# a = {'d': 20, 'f': 12, 'a': 20}
# b = a.get
# print(b('d'))

# v = [key for key, value in sorted(a.items(), key=lambda items: (-items[1], items[0]))]
# print(v)

# rows = 4
# cols = 3

# print([[0] * cols] * rows)

# a = range(1, 2)
# print(type(str))

class Mylist:
    def __init__(self, *values):
        self.values = values
        self.inx = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.inx < len(self.values):
            self.inx += 1
            return self.values[self.inx - 1]
        else:
            raise StopIteration

# for i in Mylist(31, 24, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31):
#     print(i)
# a, *b, c = (1, 5)
# print(a, b, c)
# print(slice(*[1, 5]))
# from range import Range
# r = Range(1, 10)
# a, *b, c = r
# print(a, b, c)
# print(slice(*r))
# list = [31, 24, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# print(list[12:20])
# month = 2
# print(float('-inf'))
# for i in range(2, 3):
#     print(1)

# alist = [31, 24, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# alist.insert(len(alist), 0)
# print(alist)
# x = 0
# if x:
#     print(True)

# x = 4
# # x += -1
# # print(f'{x:f}')

# # a = [('usd', 1.5), ('ud', 1.5), ('d', 1.5), ('sd', 1.5)]

# # b = {"usd": 1.75, "euro": 1.95, "gbp": 2.9}

# # def v(kwargs):
# #     a = dict(kwargs)
# #     print(kwargs)

# # v(**dict(a))
# # v(**dict(b))
# # v(a)
# # v(b)


# import numpy as np
# b = np.array([1, 2, 3], dtype='H')
# print(b)
# print(type(b[0]))

# # array([1.,  2.,  3.], dtype=float32)

# a = np.array([1, 2, 3], dtype='d')
# print(a)
# print(type(a[0]))

# # array([1.,  2.,  3.], dtype=float64)


# import array
# print(array.typecodes)

# alist = [31, 24, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# l = [30, 31, 30, 31]
# a = (31, 24, 31)
# r = (31, 24, 31)

# for v in zip(r):
#     print(v)

# s = 0
# print(hasattr(s, '__iter__'))

# s1 = {1, 3, 4}
# s2 = {1, 4, 3}

# l1 = [1, [1, 3], 3]
# l2 = [1, [1, 4], 3]
# # print(l1 == l2)

# s1 = {1: 1, (4, ): (8, 7), 3: [3]}
# s2 = {1: 1, (4, ): (8, 7), 3: [3]}
# # print(type(s1) == type(s2))
# # for a, b in zip(s1, s2):
# #     print(a, b)

# a = {(1, 1), (3, 4)}
# b = {(3, 4), (1, 2)}
# print(a)

print(bin(4))