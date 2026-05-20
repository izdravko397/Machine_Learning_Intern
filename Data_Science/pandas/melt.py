import numpy as np
from dataframe import DataFrame
from series import Series
from index import Index, RangeIndex

def melt(df: DataFrame, id_vars=None, value_vars=None):
    if value_vars is None:
        cols_inx = np.where(df.columns._data != id_vars)[0]
    else:
        cols_inx = df.columns.get_indexer(value_vars)

    new_index = RangeIndex(len(df) * len(cols_inx))
    
    if id_vars is not None:
        id_vars_ser = df[id_vars]
        id_res_ser = Series(np.tile(id_vars_ser._data, len(cols_inx)), new_index, id_vars)


    cols_vals = df.columns._data[cols_inx]
    var_ser_data = np.fromiter((c for c in cols_vals for _ in range(len(df))), dtype=df.columns._dtype)
    var_res_ser = Series(var_ser_data, new_index, name='variable')


    cols_inx_iter = iter(cols_inx)
    val_ser_data = df._series[next(cols_inx_iter)]._data
    for inx in cols_inx_iter:
        val_ser_data = np.concatenate([val_ser_data, df._series[inx]._data])

    val_res_ser = Series(val_ser_data, new_index, name='value')

    if id_vars is None:
        df_data = [var_res_ser, val_res_ser]
    else:
        df_data = [id_res_ser, var_res_ser, val_res_ser]

    return DataFrame(df_data, new_index)
    