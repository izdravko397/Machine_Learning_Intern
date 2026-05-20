from dataframe import DataFrame
from series import Series
import numpy as np

obj = Series(['s', 'a', 'c', 'b'])
print(obj.rank())

obj = Series([7, -5, 7, 4, 2, 0, 4])
print(obj.rank())

s = Series([100, 200, 100, 300, 200])
print('average')
print(s.rank())  

# 0    1.5  
# 1    3.5  
# 2    1.5  
# 3    5.0  
# 4    3.5
print('min')
print(s.rank(method="min"))

# 0    1.0  
# 1    3.0  
# 2    1.0  
# 3    5.0  
# 4    3.0  
print('max')

print(s.rank(method="max"))

# 0    2.0  
# 1    4.0  
# 2    2.0   
# 3    5.0   
# 4    4.0   
print('first')

print(s.rank(method="first"))
# 0    1.0   
# 1    3.0   
# 2    2.0   
# 3    5.0   
# 4    4.0 
print('dense')

print(s.rank(method="dense"))

# 0    1.0   
# 1    2.0   
# 2    1.0  
# 3    3.0  
# 4    2.0  