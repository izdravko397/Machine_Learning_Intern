def deepcopy(item):
    if hasattr(item, '__dict__'):
        new_class = item.__class__
        new_obj = new_class.__new__(new_class)

        for key, val in item.__dict__.items():
            setattr(new_obj, key, deepcopy(val))
        return new_obj
    
    if isinstance(item, list):
        temp_sequence = []
        for i in item:
            temp_sequence.append(deepcopy(i))
        return temp_sequence
    
    if isinstance(item, tuple):
        temp_sequence = []
        for i in item:
            temp_sequence.append(deepcopy(i))
        return tuple(temp_sequence)
    
    if isinstance(item, set):
        temp_sequence = set()
        for i in item:
            temp_sequence.add(deepcopy(i))
        return temp_sequence

    if isinstance(item, dict):
        temp_dict = {}
        for key, val in item.items():
            temp_dict[deepcopy(key)] =  deepcopy(val)
        return temp_dict
    
    return item

from dataclasses import dataclass

@dataclass
class Employee:
    name: str
    age: list

@dataclass
class Manager:
    name: Employee

@dataclass
class Intern:
    name: str

v = Employee("ivan", age=[1, 3, 4 ,5])
a = [ v, Manager(v), [1, 2, 3], Intern("stoyan"), "hello"]

b = deepcopy(a)
a[0].name = 'lala'
a[2][0] = 99
a[1].name.name = 'mitko'
a[1].name.age[0] = 'mitko'

print(b)
print(a)