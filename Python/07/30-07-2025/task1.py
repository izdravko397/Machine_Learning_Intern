from range import Range
        
def range(start, stop, step=1):
    return Range(start, stop, step)

for i in range(2, 10, 3):
   print(i)