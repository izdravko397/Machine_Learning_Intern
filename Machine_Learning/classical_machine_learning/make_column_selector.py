import pandas as pd
import numpy as np

def make_column_selector(dtype_include=None, dtype_exclude=None):
    if dtype_include is None and dtype_exclude is None:
        raise ValueError('Must input dtype_include, dtype_exclude or both')

    if dtype_include is not None and not isinstance(dtype_include, list):
        dtype_include = [dtype_include]

    if dtype_exclude is not None and not isinstance(dtype_exclude, list):
        dtype_exclude = [dtype_exclude]
        
    def selector(df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'Expects a DataFrame, not: {type(df).__name__}')
        
        result = []
        for col_name in df:
            ser_dtype = df[col_name].dtype
            
            if dtype_include is not None:
                to_add = any(np.issubdtype(ser_dtype, dtp) for dtp in dtype_include)
            else:
                to_add = True

            if dtype_exclude is not None:
                for dtp in dtype_exclude:
                    if np.issubdtype(ser_dtype, dtp):
                        to_add = False
                        break

            if to_add:
                result.append(col_name)

        return result
    return selector


def make_column_selector(dtype_include=None, dtype_exclude=None):
    if dtype_include is None and dtype_exclude is None:
        raise ValueError('Must input dtype_include, dtype_exclude or both')

    def selector(df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f'Expects a DataFrame, not: {type(df).__name__}')
        
        return df.select_dtypes(dtype_include, dtype_exclude).columns.to_list()
    return selector


df = pd.DataFrame({
    "age": [25, 32, 47, 51],
    "height": [1.75, 1.68, 1.82, 1.79],
    "name": ["Ivan", "Petya", "Georgi", "Maria"],
    "city": ["Sofia", "Plovdiv", "Varna", "Burgas"],
    "is_member": [True, False, True, False],
    "score": [88, 92, 75, 83]
})

selector = make_column_selector(dtype_include=np.number, dtype_exclude=np.floating)
print(selector(df))

from sklearn.compose import make_column_selector
selector = make_column_selector(dtype_include=np.number, dtype_exclude=np.floating)
print(selector(df))



# selector = make_column_selector(dtype_include=np.floating)
# print(selector(df))
# print(df.select_dtypes(np.floating))