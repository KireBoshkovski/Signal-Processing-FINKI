import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cmath

matplotlib.use('TkAgg')

# Дaдени се сигналите x[n]=∑(h=0)=>7 δ[n-h] и y[n]=∑(h=56)=>63 δ[n-h].
# Да се најдат нивните ДФТ, да се нацртаат и споредат нивните амплитудни и фазни спектри.

N = 64
n = np.arange(0, N)


def plot_signal(signal, title):
    plt.figure(figsize=(8, 4))
    plt.stem(n, signal)
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.title(f'Изглед на сигналот {title}')
    plt.tight_layout()


def plot_dft(signal, title):
    DFT = np.fft.fft(signal)

    # Real and Imaginary Parts
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.stem(n, DFT.real)
    plt.title(f'Реален дел од DFT на {title}')
    plt.xlabel('k')
    plt.ylabel('Re{X[k]}')

    plt.subplot(2, 1, 2)
    plt.stem(n, DFT.imag)
    plt.title(f'Имагинарен дел од DFT на {title}')
    plt.xlabel('k')
    plt.ylabel('Im{X[k]}')
    plt.tight_layout()

    # Amplitude and Phase
    amplitude = np.abs(DFT)
    phase = np.zeros(N)
    for i in range(N):
        if amplitude[i] > 1.e-8:
            phase[i] = cmath.phase(DFT[i])

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.stem(n, amplitude)
    plt.title(f'Амплитуда на DFT на {title}')
    plt.xlabel('k')
    plt.ylabel('|X[k]|')

    plt.subplot(2, 1, 2)
    plt.stem(n, phase)
    plt.title(f'Фаза на DFT на {title}')
    plt.xlabel('k')
    plt.ylabel('∠X[k] (rad)')
    plt.tight_layout()


# ==== Define signals ====
x = np.zeros(N)
x[0:8] = 1

y = np.zeros(N)
y[56:64] = 1

plot_signal(x, 'x[n]')
plot_dft(x, 'x[n]')

plt.show()


plot_signal(y, 'y[n]')
plot_dft(y, 'y[n]')
plt.show()
