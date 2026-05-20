     
#  _group_reduce axis = 1
   # axis_1_funcs = {
        #     'min': np.min,
        #     'max': np.max,
        #     'count': len,
        #     'sum': np.sum,
        #     'mean': np.mean,
        #     }
        
        # df_col_data = []
        # for g_name, inxs in self.groups.items():
        #     df_col_data.append(g_name)
        #     group_ser_data = []

        #     for row_vals in zip(*(self.df._series[i]._data for i in inxs)):
        #         row_nparr = np.array(row_vals)
        #         row_notnan = row_nparr[~pd.isna(row_nparr)]

        #         if not len(row_notnan) and kind in ('max', 'min'):
        #             val = np.nan
        #         else:
        #             val = axis_1_funcs[kind](row_notnan)

        #         group_ser_data.append(val)
            
        #     group_ser = Series(group_ser_data, index=self.df.index, name=g_name)
        #     df_data.append(group_ser)

        # if self._is_multi:
        #     df_col = MultiIndex.from_tuples(df_col_data, names=self._key_name)
        # else:
        #     df_col = Index(df_col_data, name=self._key_name)

        # return DataFrame(df_data, columns=df_col, index=self.df.index)





# series
    # def _group_reduce(self, kind):
    #     operations = {
    #         'min': np.min,
    #         'max': np.max,
    #         'count': len,
    #         'sum': np.sum,
    #         'mean': np.mean,
    #         }
        
    #     inx_data =[]
    #     ser_data = []

    #     for key, inxs in self.groups.items():
    #         inx_data.append(key)
            
    #         if kind == 'size':
    #             ser_data.append(len(inxs))
    #             continue

    #         group_vals = self.series._data[inxs]
    #         group_vals_notnan = group_vals[~pd.isna(group_vals)]

    #         if len(group_vals_notnan) == 0 and kind in ('max', 'min'):
    #             val = np.nan
    #         else:
    #             func = operations[kind]
    #             val = func(group_vals_notnan)

    #         ser_data.append(val)

    #     if self._is_multi:
    #         ser_inx = MultiIndex.from_tuples(inx_data, names=self._key_name)
    #     else:
    #         ser_inx = Index(inx_data, name=self._key_name)

    #     return Series(ser_data, index=ser_inx, name=self.series.name)








        # self._sorted_values = np.sort(self._data)
        # self._sorted_positions = []
        # v = {}

        # if self._sorted_values.ndim > 1:
        #     self._sorted_values = np.fromiter((tuple(row) for row in self._sorted_values), np.object_)
        
        # for item in self._sorted_values:
        #     v[item] = v.get(item, 0) + 1
        #     reps = v[item] - 1
        #     if isinstance(item, tuple):
        #         mathches = [[i for i, t in enumerate(self._data) if t == item]]
        #     else:
        #         mathches = np.where(self._data == item)
        #     self._sorted_positions.append(mathches[0][reps])

        # v.clear()