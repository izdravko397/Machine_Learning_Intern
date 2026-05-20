from series import Series

s = Series([100, 200, 300], index=[10, 20, 30])

# print(s.asof(25))
# print(s.asof(35))
# print(s.asof(5)) 

s = Series([100, 200, 300], index=['b', 'd', 'e'])

print(s.asof('a'))
print(s.asof('d'))
print(s.asof('z')) 
print(s.asof('1')) 
