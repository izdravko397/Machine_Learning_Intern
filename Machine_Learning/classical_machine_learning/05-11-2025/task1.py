# да се изчисли Life satisfaction за българия като се ползват
#  двата модела: LinearRegression и KNeighborsRegressor


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# print(lifesat)
# Visualize the data
# lifesat.plot(kind='scatter', grid=True,
# x="GDP per capita (USD)", y="Life satisfaction")
# plt.axis([23_500, 62_500, 4, 9])

# Select a linear model
model = LinearRegression()
# model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
# X_new = [[37_655.2]] # Cyprus' GDP per capita in 2020
# print(model.predict(X_new)) # output: [[6.30165767]]


bg_GDP = [[31718]]
print(model.predict(bg_GDP)) # LinearRegression -> 5.89918083; KNeighborsRegressor(3) -> 5.7

# plt.show()


# ------------------ TASK 2 ------------------
# да се изчисли грешката при прогнозиране на Life satisfaction 
# за всяка една от държавите в данните, като се съпоставят 
# реалната стойност с прогнозната.



# country_inx_lifesat = lifesat.set_index('Country')
# print(country_inx_lifesat)
# russ_GDP = country_inx_lifesat.loc['Russia', 'GDP per capita (USD)']
# russ_life_sat = country_inx_lifesat.loc['Russia', 'Life satisfaction']

# print('Russia predict diff:', abs(model.predict([[russ_GDP]])[0, 0] - russ_life_sat))


# values = model.predict(lifesat[['GDP per capita (USD)']].values)
# lifesat['Predicted Life S'] = values.flatten()
# lifesat['Predicted Error'] = (lifesat['Predicted Life S'] - lifesat['Life satisfaction']).abs()
# lifesat.sort_values('GDP per capita (USD)', inplace=True)
# lifesat[['Life satisfaction', 'Predicted Life S']].plot()
# print(lifesat)

# plt.show()