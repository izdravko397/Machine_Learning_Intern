class number_set:
    def __init__(self, n):
        self.n = n
        self.in_container = 1 << n + 1
        self.container = []

    def add(self, num):
        if 0 > num or num > self.n:
            raise ValueError('Number is out of range')
        
        if self.in_container >> num & 1 != 1:
            self.container.append(num)
            self.in_container |= 1 << num

    def _remove_form_cont(self, num):
        self.in_container &= ~(1 << num)
        index = self.container.index(num)
        self.container[index], self.container[-1] = self.container[-1], self.container[index]
        self.container.pop()

    def remove(self, num):
        if 0 > num or num > self.n:
            raise ValueError('Number is out of range')
        
        if self.in_container >> num & 1 == 1:
            self._remove_form_cont(num)
        else:
            raise KeyError

    def discard(self, num):
        if 0 > num or num > self.n:
            raise ValueError('Number is out of range')
        
        if self.in_container >> num & 1 == 1:
            self._remove_form_cont(num)

    def pop(self):
        if self.container:
            removed_num = self.container.pop()
            self.in_container &= ~(1 << removed_num)
            return removed_num
        else:
            raise Exception('Set is empty!')
    
    def __contains__(self, num):
        return 0 <= num <= self.n and self.in_container >> num & 1 == 1

    def __iter__(self):
        return iter(self.container)
    
    def __len__(self):
        return len(self.container)
    
    def __str__(self):
        return f'{self.container}\n{bin(self.in_container)}'

import math

class number_set:
    def __init__(self, n):
        self.n = n
        self.in_container = bytearray(math.ceil((n + 1) / 8))
        self.container = []

    def _check_num(self, num):
        if not (0 <= num <= self.n):
            raise ValueError('Number is out of range')
        
    @staticmethod
    def _get_index(num):
        byte = math.ceil(num / 8)
        bit = num % 8

        return byte, bit 

    def add(self, num):
        self._check_num(num)

        if not self.__contains__(num):
            byte, bit = number_set._get_index(num)
            self.container.append(num)
            self.in_container[byte] |= (1 << bit)

    def _remove_form_cont(self, num):
        byte, bit = number_set._get_index(num)

        self.in_container[byte] &= ~(1 << bit)
        index = self.container.index(num)
        self.container[index], self.container[-1] = self.container[-1], self.container[index]
        self.container.pop()

    def remove(self, num):
        self._check_num(num)
        
        if self.__contains__(num):
            self._remove_form_cont(num)
        else:
            raise KeyError

    def discard(self, num):
        self._check_num(num)
        
        if self.__contains__(num):
            self._remove_form_cont(num)

    def pop(self):
        if self.container:
            removed_num = self.container.pop()
            byte, bit = number_set._get_index(removed_num)
            self.in_container[byte] &= ~(1 << bit)
            return removed_num
        else:
            raise Exception('Set is empty!')
    
    def __contains__(self, num):
        byte, bit = number_set._get_index(num)
        return 0 <= num <= self.n and self.in_container[byte] >> bit & 1 == 1

    def __iter__(self):
        return iter(self.container)
    
    def __len__(self):
        return len(self.container)
    
    def __str__(self):
        return f'{self.container}'
    
ns = number_set(n=1024)
ns.add(1000)

if 1000 in ns:
  print("found")
  
ns.add(2000) # ValueError