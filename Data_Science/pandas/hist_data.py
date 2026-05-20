from cut_and_qcut import cut
import numpy as np

def hist_data(nums, bins):
    cat = cut(nums, bins)
    return cat.value_counts()

data = hist_data(np.random.standard_normal(500), bins=4)
print(data)

