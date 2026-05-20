from array import array

class Myarray(array):
    # def __setitem__(self, key, value):
    #     list = self.tolist()
    #     list[key] = value
    #     self.clear()
    #     self.extend(list)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            super().__setitem__(key, value)

        if isinstance(key, slice):
            start = 0 if key.start == None else key.start
            stop = self.itemsize if key.stop == None else key.stop

            val_inx = 0
            pop_inx = start
            for i in range(start, stop):
                if val_inx < len(value):
                    self[i] = value[val_inx]
                    val_inx += 1
                    pop_inx = i + 1
                    continue

                self.pop(pop_inx)

            for i in range(val_inx, len(value)):
                self.insert(stop, value[i])
                stop += 1

a = Myarray('i', [1, 2, 3, 4])
a[:] = [7, 8]
print(a)
# [1, 7, 8, 9, 3, 4]