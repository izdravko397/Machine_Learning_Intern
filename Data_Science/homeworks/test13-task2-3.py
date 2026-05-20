import matplotlib.pyplot as plt
import pandas as pd
from series import Series
from dataframe import DataFrame
from index import Index
import numpy as np

# fig, ax = plt.subplots(1, 2)

# data = pd.Series([10, 20, 15], index=[1, 2, 3])
# data.plot.bar(ax=ax[0])

# data = Series([10, 20, 15], index=[1, 2, 3])
# data.barplot(ax=ax[1])

# plt.show()




# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# data = pd.Series([10, 20, 15], index=[1, 2, 3])
# data.plot.bar(ax=ax[0], color='skyblue', alpha=0.7)
# ax[0].set_title("pandas.Series")

# data = Series([10, 20, 15], index=[1, 2, 3])
# data.barplot(ax=ax[1], color='skyblue', alpha=0.7)
# ax[1].set_title("MySeries")

# plt.show()



# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# data = pd.Series([10, -5, 15, -8], index=[1, 2, 3, 4])
# data.plot.bar(ax=ax[0], color='orange', alpha=0.7)
# ax[0].set_title("pandas.Series")

# data = Series([10, -5, 15, -8], index=[1, 2, 3, 4])
# data.barplot(ax=ax[1], color='orange', alpha=0.7)
# ax[1].set_title("MySeries")

# plt.show()



# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# data = pd.Series([5, 12, 9], index=["A", "B", "C"])
# data.plot.bar(ax=ax[0], color='lightgreen', alpha=0.8)
# ax[0].set_title("pandas.Series")

# data = Series([5, 12, 9], index=["A", "B", "C"])
# data.barplot(ax=ax[1], color='lightgreen', alpha=0.8)
# ax[1].set_title("MySeries")

# plt.show()



# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# data = pd.Series([0.5, 1.7, 2.3, 0.9], index=[1, 2, 3, 4])
# data.plot.bar(ax=ax[0], color='purple', alpha=0.6)
# ax[0].set_title("pandas.Series")

# data = Series([0.5, 1.7, 2.3, 0.9], index=[1, 2, 3, 4])
# data.barplot(ax=ax[1], color='purple', alpha=0.6)
# ax[1].set_title("MySeries")

# plt.show()


# fig, ax = plt.subplots(1, 2, figsize=(14, 4))

# data = pd.Series(range(1, 11), index=range(1, 11))
# data.plot.bar(ax=ax[0], color='salmon')
# ax[0].set_title("pandas.Series")

# data = Series(range(1, 11), index=range(1, 11))
# data.barplot(ax=ax[1],  color='salmon')
# ax[1].set_title("MySeries")

# plt.show()

#  ==== DataFrame ====

# df = DataFrame(np.random.uniform(size=(6, 4)),
# index=["one", "two", "three", "four", "five", "six"],
# columns=Index(["A", "B", "C", "D"], name="Genus"))



# data = np.random.uniform(size=(6, 4))

# fig, ax = plt.subplots(1, 2)
# df = pd.DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=np.arange(10, 50, 10)
# )
# df.plot.bar(ax=ax[0])


# df = DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=np.arange(10, 50, 10)
# )
# df.barplot(ax=ax[1], whiskers=False)

# plt.show()



# data = np.random.uniform(size=(6, 4))

# fig, ax = plt.subplots(1, 2)
# df = pd.DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=np.arange(10, 50, 10)
# )
# df.plot.bar(ax=ax[0], stacked=True)


# df = DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=np.arange(10, 50, 10)
# )
# df.barplot(ax=ax[1], stacked=True, whiskers=True)

# plt.show()


# data = np.random.uniform(1, 10, size=(6, 4))
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# df = pd.DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=np.arange(10, 50, 10)
# )
# df.plot.bar(ax=ax[0], stacked=True, title="Pandas stacked")

# df = DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=np.arange(10, 50, 10)
# )
# df.barplot(ax=ax[1], stacked=True, whiskers=True)
# ax[1].set_title("DataFrame stacked")

# plt.show()



# data = np.random.uniform(1, 10, size=(6, 4))

# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# df = pd.DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=["A", "B", "C", "D"]
# )
# df.plot.bar(ax=ax[0], stacked=False, title="Pandas")

# df = DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=["A", "B", "C", "D"]
# )
# df.barplot(ax=ax[1], stacked=False, whiskers=True)
# ax[1].set_title("Custom DataFrame")

# plt.show()

# data = np.random.uniform(-5, 5, size=(6, 4))
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# df = pd.DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=["W", "X", "Y", "Z"]
# )
# df.plot.bar(ax=ax[0], stacked=True, title="Pandas stacked (negatives)")

# df = DataFrame(
#     data,
#     index=np.arange(1, 7),
#     columns=["W", "X", "Y", "Z"]
# )
# df.barplot(ax=ax[1], stacked=True)
# ax[1].set_title("Custom DataFrame stacked (negatives)")

# plt.show()



df_data = np.random.standard_normal((10, 4)).cumsum(0)

df = pd.DataFrame(df_data, columns=["A", "B", "C", "D"], index=np.arange(0, 100, 10))
df.plot(subplots=True, layout=(2, 2), title='Pandas DataFrame', grid=True, sharex=True, sharey=True, rot=30)


df = DataFrame(df_data, columns=["A", "B", "C", "D"], index=np.arange(0, 100, 10))
df.plot(subplots=True, layout=(2, 2), title='My Dataframe', grid=True, sharex=True, sharey=True, rot=30)
plt.show()