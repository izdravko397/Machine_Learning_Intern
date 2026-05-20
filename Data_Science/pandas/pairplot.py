import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataframe import DataFrame


def pairplot(data: DataFrame):
    col_len = len(data.columns)
    fig, ax = plt.subplots(col_len, col_len)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i in range(col_len):
        for j in range(col_len):
            ax[i, j].grid(True)

            if i == j:
                data._series[i].plot(ax=ax[i, j], kind='hist')

                if i == col_len - 1:
                    ax[i, j].set_xlabel(data.columns[i])
            else:
                data.plot(ax=ax[i, j], kind='scatter', x=data.columns[j], y=data.columns[i], alpha=0.2)

            ax[i, j].tick_params(axis='both', length=0)

            if j > 0:
               ax[i, j].set_ylabel('')
               ax [i, j].set_yticklabels([])

            if i < col_len - 1:
                ax[i, j].set_xlabel('')
                ax [i, j].set_xticklabels([])

    ax[0, 0].set_yticklabels(ax[0, 1].get_yticks())

    return ax


macro = pd.read_csv("examples/macrodata.csv")
data = macro[["cpi", "m1", "tbilrate", "unemp"]]
trans_data = np.log(data).diff().dropna()

trans_data_my_df = DataFrame(np.array([trans_data[col].array for col in trans_data]).T,
                           columns=["cpi", "m1", "tbilrate", "unemp"])
# print(trans_data.head())
# print(trans_data_my_df)

sns.pairplot(trans_data, diag_kind="kde", plot_kws={"alpha": 0.2})

pairplot(trans_data_my_df)
plt.show()