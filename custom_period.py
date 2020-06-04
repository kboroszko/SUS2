import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft
import time


first_date = pd.to_datetime('2019-12-02 09:00:00')
last_date = pd.to_datetime('2020-02-20 11:00:00')
delta = pd.Timedelta(days=0)
def date_to_idx_abs(x):
    date = pd.to_datetime(x)
    delta = pd.Timedelta(date - first_date)
    d = delta.components.days
    h = delta.components.hours
    # if (d * 24 + h) > 1922:
    #     print(x)
    #     print(delta)
    #     print('d', d,'h', h)
    return d * 24 + h


okres=24

#%%
mySolpd = pd.read_csv('data/exemplary_solution.csv', header=None)
mySol = mySolpd.to_numpy()
#%%



def get_means(arr, okres):
    m = np.nan_to_num(np.array([np.nanmean(arr[x::okres], axis=0) for x in range(okres)])[:,0])
    return m

def predict(t_test, means):
    return means[t_test]

def eval(data, means, okres):
    y_true = np.nan_to_num(data[:,0])
    x = np.remainder(np.arange(data.shape[0]), okres)
    y_pred = means[x]
    return r2_score(y_true, y_pred)


#%%
names_np_all = np.load('names.npy',allow_pickle=True)
out_norm = np.load('out_norm.npy')
#%%

names_dict = {}
for i in range(names_np_all.shape[0]):
    host, gr = names_np_all[i]
    names_dict[host+gr] = i

def get_test_t(okres):
    return (np.arange(168) + (date_to_idx_abs('2020-02-20 12:00:00') % okres)) % okres
t_test = get_test_t(okres)


#%%
idx=7
y_train = np.nan_to_num(out_norm[idx,:,0])
means =get_means(out_norm[idx], okres)
pr = predict(t_test, means)
score = eval(out_norm[idx], means, okres)
print(score)
#%%
start = time.time()
iters_num = mySolpd.shape[0]
done = []
ommited = []
for i in range(mySolpd.shape[0]):
    a,b = mySolpd.iloc[i][:2]
    if (a+b) in names_dict:
        idx = names_dict[a+b]
        # print(a,b,names_dict[a+b])
        means =get_means(out_norm[idx])
        pr = predict(t_test, means)
        score = eval(out_norm[idx], means)
        real_mean = mySol[i,2]
        means_score = np.abs(means.mean() - real_mean)
        if(score > 0.2) and means_score < 0.1*real_mean:
            row = mySolpd.iloc[i].copy()
            row[:2] = a,b
            row[2:] = pr
            mySolpd.iloc[i] = row
            done.append(idx)
        else :
            ommited.append(idx)
    if i % 100 == 0 and i > 0:
        iters_left = iters_num - i
        elapsed = time.time() - start
        eta = (elapsed/i) * iters_left
        print(i,'eta:', eta/60. , 'mins')

print('done', len(done))
mySolpd.to_csv('out5.csv', index=False, header=False)