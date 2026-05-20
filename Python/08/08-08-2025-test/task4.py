class number_set:
    def __init__(self, *args):
        self.container = []

        for e in args:
            if e < 0 or e > 100:
                continue

            self.add(e)

    def add(self, element):
        len_container = len(self.container)

        flag = False
        if len_container % 2 != 0:
            flag = True

        for i in range(1, (len_container // 2) + 1):
            if element == self.container[i - 1] or element == self.container[-i]:
                break

            if flag:
                flag = False
                if element == self.container[len_container // 2]:
                    break
        else:
            self.container.append(element)

    def remove(self, element):
        len_container = len(self.container)

        flag = False
        if len_container % 2 != 0:
            flag = True

        for i in range(1, (len_container // 2) + 1):
            if element == self.container[i - 1]:
                index = i - 1
                break

            elif element == self.container[-i]:
                index = -i
                break

            if flag:
                flag = False
                if element == self.container[len_container // 2]:
                    index = len_container // 2
                    break
        else:
            raise ValueError
        
        self.container[index], self.container[-1] = self.container[-1], self.container[index]
        self.container.pop()

    def discard(self, element):
        len_container = len(self.container)

        flag = False
        if len_container % 2 != 0:
            flag = True

        index = None
        for i in range(1, (len_container // 2) + 1):
            if element == self.container[i - 1]:
                index = i - 1
                break

            elif element == self.container[-i]:
                index = -i
                break

            if flag:
                flag = False
                if element == self.container[len_container // 2]:
                    index = len_container // 2
                    break
        
        if index != None:
            self.container[index], self.container[-1] = self.container[-1], self.container[index]
            self.container.pop()

    def pop(self):
        return self.container.pop()
    
    def __contains__(self, element):
        len_container = len(self.container)

        flag = False
        if len_container % 2 != 0:
            flag = True

        for i in range(1, (len_container // 2) + 1):
            if element == self.container[i - 1] or element == self.container[-i]:
                return True

            if flag:
                flag = False
                if element == self.container[len_container // 2]:
                    return True
                
        return False

    def __iter__(self):
        return iter(self.container)
    
    def __len__(self):
        return len(self.container)
    
    def __str__(self):
        return f'{self.container}'

class number_set:
    def __init__(self, nums=[]):
        self.in_container = [False] * 101
        self.container = []

        for n in nums:
            self.add(n)

    def add(self, num):
        if 0 <= num <= 100 and not self.in_container[num]:
            self.container.append(num)
            self.in_container[num] = True

    def remove(self, num):
        if 0 <= num <= 100 and self.in_container[num]:
            self.in_container[num] = False
            index = self.container.index(num)
            self.container[index], self.container[-1] = self.container[-1], self.container[index]
            self.container.pop()
        else:
            raise KeyError

    def discard(self, num):
        if 0 <= num <= 100 and self.in_container[num]:
            self.in_container[num] = False
            index = self.container.index(num)
            self.container[index], self.container[-1] = self.container[-1], self.container[index]
            self.container.pop()

    def pop(self):
        if self.container:
            removed_num = self.container.pop()
            self.in_container[removed_num] = False
            return removed_num
        else:
            raise Exception('Set is empty!')
    
    def __contains__(self, num):
        return 0 <= num <= 100 and self.in_container[num]

    def __iter__(self):
        return iter(self.container)
    
    def __len__(self):
        return len(self.container)
    
    def __str__(self):
        return f'{self.container}'

l = [1 ,2, 4, 5, 6 ,7, 7]
a = number_set(l)
print(a)

a.add(3)
print(a)

a.add(7)
print(a)

a.remove(1)
print(a)

a.remove(7)
print(a)

a.remove(4)
print(a)

a.discard(6)
print(a)

a.discard(4)
print(a)

for i in a:
    print(i)

print(a.pop())
print(a)

if 4 not in a:
    print(True)

print(len(a))