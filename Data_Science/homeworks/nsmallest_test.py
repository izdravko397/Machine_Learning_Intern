from series import Series
from dataframe import DataFrame
import pandas as pd

# --------- Series ---------
s = Series([30, 10, 40, 20, 30, 70, 50, 10])
print('Series')
print(s)
print('My')
print(s.nsmallest(10, keep='first'))

print('-' * 40)
print('Original')
s = pd.Series([30, 10, 40, 20, 30, 70, 50, 10])
print(s.nsmallest(10, keep='first'))


# s = Series([30, 10, 40, 20, 30, 70, 50, 10])
# print('Series')
# print(s)
# print('My')
# print(s.nsmallest(4, keep='last'))

# print('-' * 40)
# print('Original')
# s = pd.Series([30, 10, 40, 20, 30, 70, 50, 10])
# print(s.nsmallest(4, keep='last'))


# s = Series([30, 10, 40, 20, 30, 70, 50, 10])
# print('Series')
# print(s)
# print('My')
# print(s.nsmallest(4, keep='all'))

# print('-' * 40)
# print('Original')
# s = pd.Series([30, 10, 40, 20, 30, 70, 50, 10])
# print(s.nsmallest(4, keep='all'))

# --------- DataFrame ---------

# df = DataFrame({
#     'city': ['Sofia', 'Varna', 'Plovdiv', 'Burgas', 'Ruse'],
#     'population': [1300000, 350000, 350000, 200000, 350000],
#     'area': [492, 238, 180, 253, 230]
# })
# print('=== DF ===')
# print(df)
# print('My')
# print(df.nsmallest(3, 'population', keep='all'))

# print('-' * 40)
# print('Original')
# df = pd.DataFrame({
#     'city': ['Sofia', 'Varna', 'Plovdiv', 'Burgas', 'Ruse'],
#     'population': [1300000, 350000, 350000, 200000, 350000],
#     'area': [492, 238, 180, 253, 230]
# })
# print(df.nsmallest(3, 'population', keep='all'))



# df = DataFrame({
#     'city': ['Sofia', 'Varna', 'Plovdiv', 'Burgas', 'Ruse'],
#     'population': [1300000, 350000, 350000, 200000, 350000],
#     'area': [492, 230, 180, 253, 230]
# })
# print('=== DF ===')
# print(df)
# print('My')
# print(df.nsmallest(10, ['population', 'area'], keep='all'))

# print('-' * 40)
# print('Original')
# df = pd.DataFrame({
#     'city': ['Sofia', 'Varna', 'Plovdiv', 'Burgas', 'Ruse'],
#     'population': [1300000, 350000, 350000, 200000, 350000],
#     'area': [492, 230, 180, 253, 230]
# })
# print(df.nsmallest(10, ['population', 'area'], keep='all'))