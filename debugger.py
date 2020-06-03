import pandas as pd
import numpy as np
out = pd.read_csv("out.csv", header=None)
example = pd.read_csv("./data/exemplary_solution.csv", header=None)
list = np.arange(2,170)
tmp1 = out[list]
tmp2 = example[list]
tmp = (tmp2 - tmp1)**2
big_variation= ((tmp > 1).sum(axis=1)==168)
print(big_variation)
indexes = []
for i in range (len(big_variation)):
    if(big_variation[i]):
        indexes.append(i)
indexes = np.array(indexes)
to_save = out[1][indexes]
print(out[1][indexes])
to_save.to_csv("big_variance_cases.csv", float_format='%.4f', header=False)