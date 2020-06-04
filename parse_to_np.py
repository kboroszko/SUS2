import os
import pandas as pd
import numpy as np


counter = 0
lst = []
names = []
line_counter = 0
with open("data/training_series_long.csv", "r+") as f :
    f.readline()
    lines = []
    line = f.readline()
    lines.append(line)
    last = line.split(',')
    while True :
        line_counter += 1
        line = f.readline()
        values = line.split(',')
        if values[0] != last[0] or values[1] != last[1] or not line:
            full_name = "{0},{1}".format(last[0], last[1]).replace("/", "_")

            filename = "tmp.csv"
            with open(filename, "w+") as f2:
                f2.writelines(lines)
                lines = []
            data_np = pd.read_csv(filename).to_numpy()

            if data_np.shape[0] < 1:
                print('last', last)
            else :
                lst.append(data_np[:, 2:])
                names.append(data_np[0, :2])

            counter += 1
            if counter % 100 == 0:
                print(counter)

        if not line:
            break
        lines.append(line)
        last = line.split(',')
#%%
first_date = pd.to_datetime('2019-12-02 09:00:00')
last_date = pd.to_datetime('2020-02-20 11:00:00')
delta = pd.Timedelta(days=0)
def date_to_idx_abs(x):
    date = pd.to_datetime(x)
    delta = pd.Timedelta(date - first_date)
    d = delta.components.days
    h = delta.components.hours
    if (d * 24 + h) > 1922:
        print(x)
        print(delta)
        print('d', d,'h', h)
    return d * 24 + h

#%%

out_norm = np.empty((len(lst), 1923,7),dtype=np.float)
out_norm[:]=np.nan

for i in range(len(lst)):
    tab=lst[i]
    for row in tab[:]:
        d = row[0]
        idx = date_to_idx_abs(d)
        if idx > 0:
            out_norm[i,idx,:] = row[1:].astype(np.float)
    if i % 100 == 0:
        print(i)

#%%
out = np.stack(lst, axis=0)
np.save()

dt_np = np.array(dt, dtype=np.datetime64).reshape(sh1)
np.save('all.npy', out[:,1:].astype(np.float))

#%%

names_dict = {}
for i in range(names_np_all.shape[0]):
    host, gr = names_np_all[i]
    names_dict[host+gr] = i

#%%
mySolpd = pd.read_csv('data/exemplary_solution.csv', header=None)
mySol = mySolpd.to_numpy()
#%%


t_test = (np.arange(168) + (date_to_idx_abs('2020-02-20 12:00:00') % 168)) % 168

def predict(arr):
    m = np.nan_to_num(np.array([np.nanmean(arr[x::168], axis=0) for x in range(168)])[:,0])
    return m[t_test]











#%%
for i in range(mySolpd.shape[0]):
    a,b = mySolpd.iloc[i][:2]
    if (a+b) in names_dict:
        idx = names_dict[a+b]
        # print(a,b,names_dict[a+b])
        pr =predict(out_norm[idx])
        row = mySolpd.iloc[i].copy()
        row[:2] = a,b
        row[2:] = pr
        mySolpd.iloc[i] = row
    if i % 100 == 0:
        print(i)

#%%
cnt = 0
for i in range(mySolpd.shape[0]):
    row = mySolpd.iloc[i][2:]
    me = np.mean(row)
    me2 = mySol[i,2]
    r = me - me2
    if r > me*0.05 :
        cnt += 1

#%%
mySolpd.to_csv('out4.csv', index=False, header=False)
