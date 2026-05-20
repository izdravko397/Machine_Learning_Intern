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

a = number_set(5)
a.add(4)
a.add(0)
print(a)
print()

a.remove(4)
print(a)

a.discard(3)
print(a)

a.pop()
print(a)

print(1 in a)
a.add(1)
print(1 in a)