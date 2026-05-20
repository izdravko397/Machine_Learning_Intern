from task3 import Employee

def sort(arr: list):
    if not arr:
        return 
    
    inx = 0
    for i in range(len(arr)):
        min_element = arr[i]
        flag = False

        for j in range(i + 1, len(arr)):
            if arr[j] < min_element:
                min_element = arr[j]
                inx = j
                flag = True

        if flag:
            arr[i], arr[inx] = arr[inx], arr[i]

e1 = Employee("Stoyan")
e2 = Employee("Ivan")
e3 = Employee("Petar")
e4 = Employee("Albena")

ea = [e1, e2, e3, e4]
sort(ea)
print(ea)
# Albena, Ivan, Petar, Stoyan

l = [3, 5, 1, 4, 0]
sort(l)
print(l)
# [0, 1, 3, 4, 5]