from read_csv import read_csv
from index import Index, MultiIndex, RangeIndex
import numpy as np
from dataframe import DataFrame
from melt import melt

data = read_csv("examples/macrodata.csv")
data = data.loc[:, ["year", "quarter", "realgdp", "infl", "unemp"]]
print(data.head())
print()

new_inx = data['year'].astype(str) + '-' + data['quarter'].astype(str)

data = data.reindex(columns=["realgdp", "infl", "unemp"])
data.index = Index(new_inx._data, name='date')
data.columns.name = "item"
print(data.head())
print()

#-----------------------------
# periods = pd.PeriodIndex(year=data.pop("year"),
# quarter=data.pop("quarter"),
# name="date")
# print(periods)
# data.index = periods.to_timestamp("D")
# print(data.head())
# print(data.columns)
# data = data.reindex(columns=["realgdp", "infl", "unemp"])
#-----------------------------


# long_data = (data.stack().reset_index().rename(columns={'0': "value"}))
# print(long_data[:10])
# print()

# pivoted = long_data.pivot(index="date", columns="item", values="value")
# print(pivoted.head())
# print()

# long_data["value2"] = np.random.standard_normal(len(long_data))
# print(long_data[:10])
# print()

# pivoted = long_data.pivot(index="date", columns="item")
# print(pivoted.head())
# print()

# print(pivoted["value"].head())

#======== melt =======
# df = DataFrame({"key": ["foo", "bar", "baz"],
# "A": [1, 2, 3],
# "B": [4, 5, 6],
# "C": [7, 8, 9]})

# print(df)
# print()

# melted = melt(df, id_vars="key")
# print(melted)
# print()

# reshaped = melted.pivot(index="key", columns="variable", values="value")
# print(reshaped)
# print()
# print(reshaped.reset_index())
# print()

# print(melt(df, id_vars="key", value_vars=["A", "B"]))
# print()

# print(melt(df, value_vars=["A", "B", "C"]))
# print()

# print(melt(df, value_vars=["key", "A", "B"]))
# print()






# import pandas as pd

# departments = ['HR', 'HR', 'IT', 'IT', 'IT', 'Finance', 'Finance', 'Marketing', 'Marketing', 'Sales']
# employee_ids = [101, 102, 201, 202, 203, 301, 302, 401, 402, 501]

# index = MultiIndex.from_arrays(
#     [departments, employee_ids],names=['Department', 'EmployeeID']) # ,names=['Department', 'EmployeeID']

# data = {
#     'Name': np.array(['Anna', 'Boris', 'Clara', 'Deyan', 'Eva', 'Filip', 'Gergana', 'Hristo', 'Ivana', 'Julia']),
#     'YearsExperience': np.random.randint(1, 15, size=10),
#     'PerformanceScore': np.random.randint(60, 100, size=10),
#     'Salary': np.random.randint(40000, 120000, size=10)
# }

# cols = Index(list(data.keys()), name='Category')

# np.random.seed(0)
# df = DataFrame(data, index=index, columns=cols)

# print(df)



# np.random.seed(1)

# data = np.random.randint(50, 100, size=(5, 6))

# columns = MultiIndex.from_tuples([
#     ('Math', 'Score'),
#     ('Math', 'Grade'),
#     ('English', 'Score'),
#     ('English', 'Grade'),
#     ('Science', 'Score'),
#     ('Science', 'Grade')
# ], names=['Subject', 'Detail']) # , names=['Subject', 'Detail']

# students = Index(['Anna', 'Boris', 'Clara', 'Deyan', 'Eva'], name='students') # , name='students'

# df = DataFrame(data, index=students, columns=columns)

# print(df)



# departments = ['HR', 'HR', 'IT', 'IT', 'Finance', 'Finance']
# employees = [101, 102, 201, 202, 301, 302]

# index = MultiIndex.from_arrays(
#     [departments, employees],
#     names=['Department', 'EmployeeID']
# )

# # index = MultiIndex.from_arrays(
# #     [departments, employees])

# quarters = ['Q1', 'Q1', 'Q2', 'Q2']
# metrics = ['Performance', 'Salary', 'Performance', 'Salary']

# columns = MultiIndex.from_arrays(
#     [quarters, metrics],
#     names=['Quarter', 'Metric']
# )

# # columns = MultiIndex.from_arrays(
# #     [quarters, metrics])

# data = np.random.randint(60, 120, size=(len(index), len(columns)))
# df = DataFrame(data, index=index, columns=columns)

# print(df)



# data = {
#     'Name': ['Anna', 'Boris', 'Clara', 'Deyan', 'Eva'],
#     'Age': [25, 32, 29, 41, 23],
#     'Department': ['HR', 'IT', 'Finance', 'IT', 'Marketing'],
#     'Salary': [48000, 62000, 54000, 75000, 50000]
# }

# inx = RangeIndex(5) # , name='Key'
# cols = Index(list(data.keys())) # , name='Category'

# df = DataFrame(data, index=inx, columns=cols)

# print(df)