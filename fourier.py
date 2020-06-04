import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft

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


def get_means(arr):
    m = np.nan_to_num(np.array([np.nanmean(arr[x::168], axis=0) for x in range(168)])[:,0])
    return m

def predict(t_test, means):
    return means[t_test]

def eval(data, means):
    y_true = np.nan_to_num(data[:,0])
    x = np.remainder(np.arange(data.shape[0]), 168)
    y_pred = means[x]
    return r2_score(y_true, y_pred)


names_np_all = np.load('names.npy',allow_pickle=True)
out_norm = np.load('out_norm.npy')

names_dict = {}
for i in range(names_np_all.shape[0]):
    host, gr = names_np_all[i]
    names_dict[host+gr] = i

t_test = (np.arange(168) + (date_to_idx_abs('2020-02-20 12:00:00') % 168)) % 168

#%%
a,b = names_np_all[0]
idx = names_dict[a+b]

y_train = np.nan_to_num(out_norm[7,:,0])
#%%
N = y_train.shape[0]
ffy = fft(y_train)
tops = np.abs(ffy[:ffy.shape[0]//2]).argsort()[-6:-1][::-1]
T = (1./np.linspace(0, 1. / N, N))/N
plt.plot(T[:N//2], np.abs(ffy)[:N//2])
f = T[tops]
w = np.abs(ffy)[tops]
print(f)
#%%
min_f = int(f[0])
max_idx = 40*24
size = max_idx//min_f + 1
multis = np.stack([f for _ in range(size)], axis=0)
errs = np.array([((np.abs(multis-i)/i).min(axis=0)*w).mean() for i in np.arange(min_f,max_idx,min_f)])
plt.plot(np.arange(min_f,max_idx,min_f), errs)
okres = (errs.argmin()+1)*min_f
print(okres)
#%%









