import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# fl = ('data/splits/host0003-cpu_5m.csv')


def date_to_idx(x):
    date = pd.to_datetime(x)
    d = date.weekday()
    h = date.hour
    return d * 24 + h


def get_medians(data_np):
    time = data_np[:,2].astype(np.datetime64)
    data = data_np[:,3]

    preds = [[] for _ in range(24*7)]

    for i in range(data.shape[0]):
        index = date_to_idx(time[i])
        preds[index].append(data[i])

    return [np.median(x) for x in preds]

def get_means(data_np):
    time = data_np[:,2].astype(np.datetime64)
    data = data_np[:,3]

    preds = [[] for _ in range(24*7)]

    for i in range(data.shape[0]):
        index = date_to_idx(time[i])
        preds[index].append(data[i])

    return [np.mean(x) for x in preds]

def predict(t_test, means):
    cpu_preds = np.zeros(t_test.shape)
    for i in range(t_test.shape[0]):
        index = date_to_idx(t_test[i])
        cpu_preds[i] = means[index]
    return cpu_preds

#%%


#
# cpu_pred = predict(t_test)
# plt.plot(t_test, cpu_test, label='test')
# plt.plot(t_test, cpu_pred, label='prediction')
# plt.legend()

#%%
# import sklearn.metrics as metrics
# metrics.r2_score(cpu_test, cpu_pred)

#%%

# data_np = pd.read_csv(fl).to_numpy()
# m1 = get_means(data_np)
# m2 = get_medians(data_np)
start_date = pd.to_datetime('2020-02-20 12:00:00')
t_test = np.array([start_date + pd.to_timedelta(str(x)+"h") for x in range(168)])
# preds1 = predict(t_test, m1)
# preds2 = predict(t_test, m2)
#
# plt.plot(t_test, preds1, label='means')
# plt.plot(t_test, preds2, label='medians')
# plt.legend()
#%%
trueVal = pd.read_csv('data/exemplary_solution.csv', header=None).to_numpy()[:,2:]
solVal = pd.read_csv('out2.csv', header=None).to_numpy()[:,2:]
#%%
from sklearn.metrics import r2_score

scores = np.array([r2_score(solVal[i], trueVal[i]) for i in range(trueVal.shape[0])])
means = np.array([solVal[i].mean() for i in range(trueVal.shape[0])])
