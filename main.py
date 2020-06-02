import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#%%

solution = pd.read_csv('data/exemplary_solution.csv')

#%%
template = pd.read_csv('data/solution_template.csv')
#%%

training = pd.read_csv('data/training_series_short.csv')

#%%

cpu = pd.read_csv('data/splits/host0001-cpuusagebyproc.csv').to_numpy()
# err_in = pd.read_csv('data/splits/host0001-error_in.csv').to_numpy()
# err_out = pd.read_csv('data/splits/host0001-error_out.csv').to_numpy()
# mem = pd.read_csv('data/splits/host0001-memoryallocatedbyproc.csv').to_numpy()
cpu_1m = pd.read_csv('data/splits/host0003-cpu_1m.csv').to_numpy()
cpu_5m = pd.read_csv('data/splits/host0003-cpu_5m.csv').to_numpy()
cpu_5s = pd.read_csv('data/splits/host0003-cpu_5s.csv').to_numpy()



#%%
plt.figure(figsize=(15,4))
time = cpu[:,2].astype(np.datetime64)

#%%
# plt.plot(time[:500], err_in[:500,3])
# plt.plot(time[:500], err_in[:500,4])
# plt.plot(time[:500], err_in[:500,5])
# plt.plot(time[:500], err_in[:500,6])
# plt.plot(time[:500], err_in[:500,7])
# plt.plot(time[:500], err_in[:500,8])
# plt.plot(time[:500], err_in[:500,9])

# plt.plot(time[:500], cpu_5s[:500,3], label='5s')
# plt.plot(time[:500], cpu_1m[:500,3],label='1m')
# plt.plot(time[:500], cpu_5m[:500,3], label='5m')
plt.plot(time[:500], cpu[:500,3])
# plt.plot(time[:500], err_in[:500,9])
# plt.plot(time[:500], tnp[:500,3])

plt.show()

#%%

cpu_train = cpu[:1000,3]
t_train = time[:1000]
cpu_test = cpu[1000:1300,3]
t_test = cpu[1000:1300]

#%%
plt.plot(t_train, cpu_train)
#%%
def get_best_peaks(X, peaks, k=3):
    vals = X[peaks]
    idxs = np.argsort(vals)[::-1]
    max_idx = min(k, idxs.shape[0])
    return peaks[idxs[:max_idx]]





#%%
from scipy.fft import fft,ifft
from scipy.signal import find_peaks
ffcpu = fft(cpu_train)
# ffcpu[ffcpu.shape[0]//2:] = 0
reffcpu = np.zeros(ffcpu.shape)

distance=20

peaks = find_peaks(np.abs(ffcpu), distance=distance)
beast_peaks = get_best_peaks(np.abs(ffcpu),peaks[0], k=6)


plt.plot(np.abs(ffcpu[:ffcpu.shape[0]//2]))
for pik in beast_peaks:
    fr = max(0, pik-distance)
    to = min(pik+distance, ffcpu.shape[0])
    reffcpu[fr:to] = ffcpu[fr:to]
plt.figure()
plt.plot(np.abs(reffcpu))
#%%
ifcpu = ifft(reffcpu)
plt.figure()
plt.plot(np.abs(ifcpu), label='refined')
plt.plot(cpu_train, label='true')
plt.legend()

#%%

bigger = np.zeros(cpu_train.shape[0]*2)
bigger[:cpu_train.shape[0]] = cpu_train[:]

ffbig = fft(bigger)
# ffbig[ffbig.shape[0]//2:] = 0

plt.plot(np.abs(ffbig[:ffbig.shape[0]//2]))

#%%
from skimage.transform import resize
ffcpu_stretched_r = resize(ffcpu.real, bigger.shape)
ffcpu_stretched_i = resize(ffcpu.imag, bigger.shape)
ffcpu_stretched = ffcpu_stretched_r + ffcpu_stretched_i*1j

plt.plot(np.abs(ffcpu_stretched[:ffbig.shape[0]]))
#%%
plt.figure()
iffbig = ifft(ffcpu_stretched)
plt.plot(np.abs(iffbig))
plt.plot(bigger)
