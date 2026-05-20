class Employee:
    def __init__(self, name):
        self.name = name
    
    def __lt__(self, other):
        # return self.name < other.name

        for x, y in zip(self.name, other.name):
            if ord(x) == ord(y):
                continue
            return ord(x) < ord(y)
        return len(self.name) < len(other.name)
    
    def __repr__(self):
        return f'{self.name}'
    
e1 = Employee("Stoyan")
e2 = Employee("Ivan")
e3 = Employee("Petar")
e4 = Employee("Albena")
e5 = Employee("Alben")

ea = [e1, e2, e3, e4, e5]

print(sorted(ea, key=lambda x: x.name))
# // Albena, Ivan, Petar, Stoyan