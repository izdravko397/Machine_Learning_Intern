from series import Series
from index import Index
from dataframe import DataFrame
import numpy as np
import matplotlib.pyplot as plt

s = Series(np.random.standard_normal(10).cumsum(), index=np.arange(0, 100, 10))
plot = s.plot()

# fig, axes = plt.subplots(2, 1)
# data = Series(np.random.uniform(size=16), index=list("abcdefghijklmnop"))
# data.plot(ax=axes[0], kind='bar', color="black", alpha=0.7)
# data.plot(ax=axes[1], kind='barh', color="black", alpha=0.7)

# plt.show()



# data = [1, 3, 2, 5, 4]
# s = Series(data, index=[10, 20, 30, 40, 50])


# ax1 = s.plot(kind='line')
# ax2 = s.plot(kind='line', style='r--', alpha=0.5, label='line test')

# ax3 = s.plot(kind='bar', alpha=0.6, label='bar test', title='Bar Chart')

# ax4 = s.plot(kind='barh', alpha=0.7, label='barh test')

# fig, ax = plt.subplots(1, 2)
# data = np.random.standard_normal(100)
# ax[0].hist(data, bins=20, color="black", alpha=0.3)
# s = Series(data)

# ax5 = s.plot(ax=ax[1], kind='hist', alpha=0.5, bins=20, label='hist test')

# ax6 = s.plot(kind='scatter', alpha=0.8, label='scatter test')

# ax7 = s.plot(kind='line', use_index=False, label='no index')

# s_log = Series([1, 10, 100, 1000])
# ax8 = s_log.plot(kind='line', logy=True, label='log y')

# ax9 = s.plot(xticks=[0,1,2,3,4], yticks=[1,2,3,4,5], use_index=False, rot=45,label='test', xlim=(0,4), ylim=(0,6), grid=True)

# fig, ax_custom = plt.subplots()
# ax10 = s.plot(ax=ax_custom, label='custom ax')


# s.plot(kind='invalid')
# ax = s.plot()
# ax = s.plot(figsize=(3, 3))



df = DataFrame(np.random.uniform(size=(6, 4)),
index=["one", "two", "three", "four", "five", "six"],
columns=Index(["A", "B", "C", "D"], name="Genus"))
# df.plot(kind='bar')


import pandas as pd
df_data = np.random.standard_normal((10, 4)).cumsum(0)
# print(df_data)

df = pd.DataFrame(df_data, columns=["A", "B", "C", "D"], index=np.arange(0, 100, 10))
# df = pd.DataFrame({'values': np.random.randn(1000)})
# fig, ax = plt.subplots(1, 2)

# df.plot(ax=ax[0], kind='hist', bins=50)
# df.plot(ax=ax[0])
# df.plot(ax=ax[0], kind='bar')
# df.plot(ax=ax[0], kind='barh')

df = DataFrame(df_data, columns=["A", "B", "C", "D"], index=np.arange(0, 100, 10))
# df = DataFrame({'values': np.random.randn(1000)})

# df.plot(ax=ax[1], kind='hist', bins=50)
# df.plot(ax=ax[1])
# df.plot(ax=ax[1], kind='bar')
# df.plot(ax=ax[1], kind='barh', title='custom', xlim=[-2,2])




df_data = np.random.standard_normal((10, 4)).cumsum(0)

df = DataFrame(df_data, columns=["A", "B", "C", "D"], index=np.arange(0, 100, 10))

# df.plot(ax=ax[1], kind='hist', bins=50)
df.plot(subplots=True, layout=(2, 2), title='A', grid=True, sharex=True, sharey=True, rot=30)
# df.plot(ax=ax[1], kind='bar')
# df.plot(ax=ax[1], kind='barh')



# df_data = np.random.standard_normal((10, 2)).cumsum(0)
# df = pd.DataFrame(df_data, columns=["A", "B"], index=np.arange(0, 100, 10))

# fig, ax = plt.subplots(1, 2)
# # df.plot(ax=ax[0], kind='scatter')
# df.plot(kind='scatter', x='A', y='B', ax=ax[0], title='A vs B')

# df = DataFrame(df_data, columns=["A", "B"], index=np.arange(0, 100, 10))
# df.plot(ax=ax[1], kind='scatter', x='A', y='B', title='A vs B')



plt.show()