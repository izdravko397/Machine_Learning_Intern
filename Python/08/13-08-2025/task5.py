class my_list(list):
    def __getitem__(self, key):
        val = super().__getitem__(key)

        if not isinstance(key, slice):
            return val

        for i in iter(val):
            yield i
            

# class Lazy_seqence:
#     def __init__(self, arr):
#         self.arr = arr

#     def __iter__(self):
#         for el in iter(self.arr):
#             yield el

# class my_list(list):
#     def __getitem__(self, key):
#         res = super().__getitem__(key)

#         if isinstance(key, slice):
#             return Lazy_seqence(res)
#         return res
  
arr = my_list([1, 2, 3, 4, 5])

for e in arr[1:3]:
  print(e)

print(arr[0])
# 2
# 3