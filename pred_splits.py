from avg_pred import *
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

tables = []

with open('data/solution_template.csv') as f :
    line = f.readline()
    while line:
        splits = line.split(',')
        tables.append("{0},{1}".format(splits[0], splits[1]).replace("/", "_"))
        line = f.readline()
#%%
print(tables[0])

#%%

with open("out.csv", "w+") as f:
    pass

t1 = time.time()
dt_avg = 0
for i in range(len(tables)):
    line = tables[i]
    splited_line = line.split(",")
    filename = "./data/splits/" + splited_line[0] + "-" + splited_line[1] + ".csv"
    data_np = pd.read_csv(filename).to_numpy()
    m = get_means(data_np)
    preds1 = predict(t_test, m)
    out_file = "out.csv"
    with open(out_file, "a") as f2:
        f2.write("{0},{1},".format(splited_line[0], splited_line[1]))
        for v in preds1:
            f2.write("{0},".format(v))
        f2.write("\n")
    if i % 10 == 0:
        t2 = time.time()
        dt = t2 - t1
        dt_avg = 0.8 * dt_avg + 0.2 * dt
        t1 = t2
    if i % 100 == 0:
        left = 10000 - i
        eta = left * dt_avg
        print(i, "eta:", eta / 60., 'min')
