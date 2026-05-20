from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

def cost_function_RMSE(real_data, predicted_data):
    square_diffs = np.square(real_data - predicted_data)
    return np.sqrt(np.mean(square_diffs))

def cost_function_MAE(real_data, predicted_data):
    diffs = np.abs(real_data - predicted_data)
    return np.mean(diffs)


data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
# print(lifesat)

X = lifesat[['GDP per capita (USD)']].values
y = lifesat[['Life satisfaction']].values

test_set_len = int(0.2 * len(X))

RMSE_errors_linear_test, MAE_errors_linear_test = [], []
RMSE_errors_kneighbors_test, MAE_errors_kneighbors_test = [], []

RMSE_errors_linear_train, MAE_errors_linear_train = [], []
RMSE_errors_kneighbors_train, MAE_errors_kneighbors_train = [], []

for _ in range(len(X) // test_set_len):
    X_train = X[:-test_set_len]
    y_train = y[:-test_set_len]

    X_test = X[-test_set_len:]
    y_test = y[-test_set_len:]

    linear_model = LinearRegression()
    kneighbors_model = KNeighborsRegressor(3)

    linear_model.fit(X_train, y_train)
    kneighbors_model.fit(X_train, y_train)

    linear_model_predicts_test = linear_model.predict(X_test)
    kneighbors_model_predicts_test = kneighbors_model.predict(X_test)
    RMSE_errors_linear_test.append(cost_function_RMSE(y_test, linear_model_predicts_test))
    MAE_errors_linear_test.append(cost_function_MAE(y_test, linear_model_predicts_test))
    RMSE_errors_kneighbors_test.append(cost_function_RMSE(y_test, kneighbors_model_predicts_test))
    MAE_errors_kneighbors_test.append(cost_function_MAE(y_test, kneighbors_model_predicts_test))


    linear_model_predicts_train = linear_model.predict(X_train)
    kneighbors_model_predicts_train = kneighbors_model.predict(X_train)
    RMSE_errors_linear_train.append(cost_function_RMSE(y_train, linear_model_predicts_train))
    MAE_errors_linear_train.append(cost_function_MAE(y_train, linear_model_predicts_train))
    RMSE_errors_kneighbors_train.append(cost_function_RMSE(y_train, kneighbors_model_predicts_train))
    MAE_errors_kneighbors_train.append(cost_function_MAE(y_train, kneighbors_model_predicts_train))

    X = np.roll(X, test_set_len)
    y = np.roll(y, test_set_len)


print(f'Linear model mean RMSE train data: {np.mean(RMSE_errors_linear_train):.3f}')
print(f'Linear model mean MAE train data: {np.mean(MAE_errors_linear_train):.3f}')
print('\n')

print(f'Kneighbors model mean RMSE train data: {np.mean(RMSE_errors_kneighbors_train):.3f}')
print(f'Kneighbors model mean MAE train data: {np.mean(MAE_errors_kneighbors_train):.3f}')

print('\n\n')

print(f'Linear model mean RMSE test data: {np.mean(RMSE_errors_linear_test):.3f}')
print(f'Linear model mean MAE test data: {np.mean(MAE_errors_linear_test):.3f}')
print('\n')

print(f'Kneighbors model mean RMSE test data: {np.mean(RMSE_errors_kneighbors_test):.3f}')
print(f'Kneighbors model mean MAE test data: {np.mean(MAE_errors_kneighbors_test):.3f}')


# кой модел генерализира по-добре? => според мен би бил LinearRegression,
#  но резултатите в конкретния пример и конкретните данни показват че KNeighborsRegressor е по добър