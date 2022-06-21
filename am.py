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
m = resample(m,a*len(m))
ts = 1/fs
fc = 8e3 # 


N = len(m)
T = N*ts
t = arange(0, T, ts)
c = cos(2* pi * fc * t)

Ac, ka = 1, 0.75;
noise = 0* randn(N)
m = m +noise
am = Ac*(1 + ka * m) * c

fig1 = figure(1, figsize = (12, 12))
subplot(211)
plot(t,c)
grid(True)
plot(t, m, 'r')
subplot(212)
plot(t,am)
grid(True)
fig1.savefig('1', dpi = 300)

fig2 = figure('1', figsize =(12,12))
subplot(211)
f = (arange(N) - N/2) * fs/N
plot(f, 10 * log10(abs(fftshift(fft(m)))));
subplot(212)
plot(f, 10 * log10(abs(fftshift(fft(am)))));
fig2.savefig('2', dpi = 300 )

tau = 4e-3
deAm = RC(am, ts, tau)
# deAm = am * c
b, a = butter(5, 0.1, 'low')
deAm = filtfilt(b, a, deAm)

fig3 = figure(3, figsize = (12,12))
plot(deAm)
grid(True)
fig3.savefig('3', dpi = 300)

write('AM_without_noise.wav', fs, deAm)