import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import cmath

from scipy.fft import fft

matplotlib.use('TkAgg')


def custom_dft_matrix(x):
    N = len(x)
    n = np.arange(N)
    m = n.reshape((N, 1))  # transponira

    W = np.exp(-2j * np.pi * n * m / N)  # generiranje na kolonata
    X = np.dot(W, x)  # X = Wx

    return X


def custom_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)

    return X


n = 64
N = np.linspace(0, n-1, num=n, dtype=int)
print("N: ", N)
# n = np.asarray(list(range(0, N)))  # од 0 до 63
Omega = (2.0 * np.pi / 64) * 2
Fi = np.pi / 6
# x = np.zeros(N)
# for i in range(N):
#     # x[i] = np.cos(Omega * i + Fi)
#     x[i] = 2 * np.cos(Omega * i + Fi) + np.sin(Omega * 3 * i) + np.cos(Omega * 5 * i + np.pi / 3)
x = 2 * np.cos(Omega * N + Fi) + np.sin(Omega * 3 * N) + np.cos(Omega * 5 * N + np.pi / 3)

DFT = fft(x)
print("DFT output: ", DFT)
C_DFT = custom_dft_matrix(x)
print("C_DFT output: ", C_DFT)


plt.figure("Споредба на реален дел", figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(N, C_DFT.real, label='Custom DFT (Real)')
plt.subplot(1, 2, 2)
plt.stem(N, DFT.real, label='FFT (Real)')
plt.show()

plt.figure("Споредба на имагинарен дел", figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(N, C_DFT.imag, label='Custom DFT (Imag)')
plt.subplot(1, 2, 2)
plt.stem(N, DFT.imag, label='FFT (Imag)')
plt.show()

# Амплитуда и фаза
Faza = x
C_Faza = x
for i in range(n):
    if abs(DFT[i]) > 1.e-8:
        Faza[i] = cmath.phase(DFT[i])
    if abs(C_DFT[i]) > 1.e-8:
        C_Faza[i] = cmath.phase(C_DFT[i])

plt.figure("Амплитуда", figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(N, abs(DFT))
plt.subplot(1, 2, 2)
plt.stem(N, abs(C_DFT))
plt.show()

plt.figure("Фаза", figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(N, Faza)
plt.subplot(1, 2, 2)
plt.stem(N, C_Faza)
plt.show()
