# да се разработу функция max която получава масив с обекти и връща максималния от тях. 
# функцията да има незадъжлителен параметър compare - lambda функция с която да сравняваме елементите.

def max(arr: list, compare=lambda x, y: x > y):
    if not arr:
        return
    
    max_element = arr[0]
    for i in range(1, len(arr)):
        item = arr[i]
        if compare(item, max_element):
            max_element = item

    return max_element

from task1 import dataclass_decorator

@dataclass_decorator
class Employee:
    name: str
    age: int

people = [
    Employee("Ivan", 25),
    Employee("Niki", 33),
    Employee("Desi", 12) 
]

oldest = max(people, lambda p1, p2: p1.age > p2.age)
print(oldest) # Niki

youngest = max(people, lambda p1, p2: p1.age < p2.age)
print(youngest) # Desi