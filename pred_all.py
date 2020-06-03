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

with open("out.csv", "w+") as f:
    pass
#%%
tables_idx = 0
counter = 0
start = time.time()
with open("data/training_series_long.csv", "r+") as f :
    f.readline()
    lines = []
    line = f.readline()
    lines.append(line)
    last = line.split(',')
    t1 = time.time()
    dt_avg = 0
    while True :
        line = f.readline()

        values = line.split(',')
        if values[0] != last[0] or values[1] != last[1] or not line:
            full_name = "{0},{1}".format(last[0], last[1]).replace("/", "_")
            if full_name == tables[tables_idx]:
                tables_idx += 1
                filename = "tmp.csv"
                with open(filename, "w+") as f2:
                    f2.writelines(lines)
                    lines = []
                data_np = pd.read_csv(filename).to_numpy()
                m = get_means(data_np)
                preds1 = predict(t_test, m)
                out_file = "out.csv"
                with open(out_file, "a") as f2:
                    f2.write("{0},{1},".format(last[0], last[1]))
                    for v in preds1:
                        f2.write("{0},".format(v))
                    f2.write("\n")
                counter += 1
                if counter % 10 == 0:
                    t2 = time.time()
                    dt = t2 - t1
                    dt_avg = 0.8*dt_avg + 0.2*dt
                    t1 = t2
                if counter % 100 == 0:
                    left = 10000 - counter
                    eta = left * dt_avg
                    print(counter, "eta:", eta/60., 'min')

        if not line:
            break
        lines.append(line)
        last = line.split(',')