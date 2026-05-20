from index import *
from dataframe import DataFrame

def to_numeric(arr):
    for i, item in enumerate(arr):
        try:
            arr[i] = float(item) if '.' in item else int(item)
        except:
            pass

def read_table(fname, sep='\t', header=None, names=None):
    if names is not None:
        names_len = len(names)
    
    first_row_data = None
    n_cols = None
    def data_gen():
        nonlocal n_cols, first_row_data

        with open(fname) as file:
            for line in file:
                row_data = line[:-1].split(sep)
                to_numeric(row_data)
                row_data_len = len(row_data)

                if n_cols is None:
                    first_row_data = row_data
                    n_cols = row_data_len

                if (names is not None and names_len != row_data_len) or n_cols != row_data_len:
                    raise ValueError('Different lengths on names and row data')
                
                yield row_data

    new_data = np.fromiter(data_gen(), np.object_)

    if names is not None:
        columns = Index(names)
    elif header is not None:
        columns = Index(first_row_data)
    else:
        columns = RangeIndex(n_cols)

    return DataFrame(new_data, columns=columns)


# print(read_table("examples/movies.dat", '::').head())
                
