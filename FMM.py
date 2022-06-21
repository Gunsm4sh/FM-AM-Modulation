
import numpy as np
from matplotlib.pyplot import plot, grid, figure, subplot
from numpy.fft import fft, fftshift
from numpy import cos, pi, zeros, log10, exp, arange, mean

from numpy.random import randn
from scipy.io.wavfile import read, write
from scipy.signal import butter, filtfilt, resample
import matplotlib.pyplot as plt


plt.rcParams['agg.path.chunksize'] = 10000

def RC(x, ts, tau):
    N = len(x)
    y, vc = zeros(N), zeros(N)
    y[0] = max(0, x[0])
    vc[0] = y[0]
    for i in range(1, N):
        vc[i] = vc[i - 1] * exp(-ts/tau)
        y[i] = max(vc[i], x[i])
        vc[i] = y[i]
    return y


fs, m = read('learning.wav')
m = m - mean(m)
m = m/max(abs(m))
a = 8
fs = fs * a
m = resample(m, a * len(m))
ts = 1/fs
fc = 15000

Ac, kf = 1, 4000;

N = len(m)
T = N * ts
t = arange(0, T, ts)


Ac, ka, kf = 1, 0.75, 4000;
noise = 0 * randn(N)
m = m +noise
#am = Ac * (1 + ka * m) * c
fm = Ac * cos (2 * pi * fc * t + m*np.sin(2*pi*kf*t)) # the equation for fm modulation


fig1 = figure('1', figsize =(12,12))
subplot(211)
f = (arange(N) - N/2) * fs/N
plot(f, 10 * log10(abs(fftshift(fft(m)))));
subplot(212)
plot(f, 10 * log10(abs(fftshift(fft(fm)))));
fig1.savefig('2', dpi = 300 )

tpfm = np.diff(fm)
tau = 5e-5 # modify tau to get optimum value
tp1 = RC(-tpfm, ts, tau)
tp2 = RC(tpfm, ts, tau)
deFm = tp1 + tp2
b, a = butter(9, 0.1, 'low') #tune the low pass filter
deFm = filtfilt(b, a, deFm)


fig2 = figure(3, figsize = (12,12))
plot(deFm)
grid(True)
fig2.savefig('3', dpi = 300)


write('testFm_without_Noise.wav', fs, deFm)
