from avg_pred import *

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

tables = []

with open('data/solution_template.csv') as f :
    line = f.readline()
    while line:
        splits = line.split(',')
        tables.append("{0},{1}".format(splits[0], splits[1]))
        line = f.readline()


with open("out.csv", "w+") as f:
    pass


counter = 0
with open("data/training_series_long.csv", "r+") as f :
    f.readline()
    lines = []
    line = f.readline()
    lines.append(line)
    last = line.split(',')
    while True :
        line = f.readline()

        if not line:
            break
        values = line.split(',')
        if values[0] != last[0] or values[1] != last[1] :
            full_name = "{0},{1}".format(last[0], last[1]).replace("/", "_")
            if full_name in tables :
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
                print(counter)
                counter += 1
            else :
                print(full_name, "not in solution template, ommiting...")
        lines.append(line)
        last = line.split(',')