from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
print(lifesat)


X = lifesat[['GDP per capita (USD)']].values
y = lifesat[['Life satisfaction']].values

inxs = np.random.permutation(len(X))
X = X[inxs]
y = y[inxs]

train_part_inx = int(0.8 * len(X))

X_train = X[:train_part_inx]
y_train = y[:train_part_inx]

X_test = X[train_part_inx:]
y_test = y[train_part_inx:]

linear_model = LinearRegression()
kneighbors_model = KNeighborsRegressor(3)

linear_model.fit(X_train, y_train)
kneighbors_model.fit(X_train, y_train)

def cost_function_RMSE(real_data, predicted_data):
    square_diffs = np.square(real_data - predicted_data)
    return np.sqrt(np.mean(square_diffs))

def cost_function_MAE(real_data, predicted_data):
    diffs = np.abs(real_data - predicted_data)
    return np.mean(diffs)

fig, ax = plt.subplots(2, 1)
# ======================== TEST DATA ========================
results_data_from_tests_linear = linear_model.predict(X_test).flatten()
results_data_from_tests_kneighbors = kneighbors_model.predict(X_test).flatten()

y_test = y_test.flatten()
test_data_res_table = pd.DataFrame({
    'Real Value': y_test.flatten(),
    'Linear Model Predicts': results_data_from_tests_linear,
    'KNeighbors Model Predicts': results_data_from_tests_kneighbors,
    'Linear Model Diff': np.abs(y_test - results_data_from_tests_linear),
    'KNeighbors Model Diff': np.abs(y_test - results_data_from_tests_kneighbors)
})
print('------------------- Results From TEST data -------------------')
print(test_data_res_table)
print('\n\n\n')
print('Cost Function RMSE for LinearRegression:', cost_function_RMSE(y_test, results_data_from_tests_linear))
print('Cost Function MAE for LinearRegression:', cost_function_MAE(y_test, results_data_from_tests_linear))

print('Cost Function RMSE for KNeighborsRegressor:', cost_function_RMSE(y_test, results_data_from_tests_kneighbors))
print('Cost Function MAE for KNeighborsRegressor:', cost_function_MAE(y_test, results_data_from_tests_kneighbors))
print('\n\n\n')
test_data_res_table.plot(ax=ax[0], title='Test data')

# ======================== TRAIN DATA ========================
results_data_from_train_linear = linear_model.predict(X_train).flatten()
results_data_from_train_kneighbors = kneighbors_model.predict(X_train).flatten()

y_train = y_train.flatten()
train_data_res_table = pd.DataFrame({
    'Real Value': y_train.flatten(),
    'Linear Model Predicts': results_data_from_train_linear,
    'KNeighbors Model Predicts': results_data_from_train_kneighbors,
    'Linear Model Diff': np.abs(y_train - results_data_from_train_linear),
    'KNeighbors Model Diff': np.abs(y_train - results_data_from_train_kneighbors)
})
print('------------------- Results From TRAIN data -------------------')
print(train_data_res_table)
print('\n\n\n')

print('Cost Function RMSE for LinearRegression:', cost_function_RMSE(y_train, results_data_from_train_linear))
print('Cost Function MAE for LinearRegression:', cost_function_MAE(y_train, results_data_from_train_linear))

print('Cost Function RMSE for KNeighborsRegressor:', cost_function_RMSE(y_train, results_data_from_train_kneighbors))
print('Cost Function MAE for KNeighborsRegressor:', cost_function_MAE(y_train, results_data_from_train_kneighbors))
train_data_res_table.plot(ax=ax[1], title='Train data')

df = pd.DataFrame({'Linear Model Test': [cost_function_RMSE(y_test, results_data_from_tests_linear), cost_function_MAE(y_test, results_data_from_tests_linear)], 'Linear Model Train': [cost_function_RMSE(y_train, results_data_from_train_linear), cost_function_MAE(y_train, results_data_from_train_linear)],
                    'KNeighbors Test': [cost_function_RMSE(y_test, results_data_from_tests_kneighbors), cost_function_MAE(y_test, results_data_from_tests_kneighbors)], 'KNeighbors Train': [cost_function_RMSE(y_train, results_data_from_train_kneighbors), cost_function_MAE(y_train, results_data_from_train_kneighbors)]}, index=['RMSE', 'MAE'])
print(df.T)
# plt.show()

# кой модел генерализира по-добре? => според мен би бил LinearRegression,
#  но резултатите в конкретния пример и конкретните данни показват че KNeighborsRegressor е по добър