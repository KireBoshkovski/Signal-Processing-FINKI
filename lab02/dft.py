import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

omega = np.linspace(-np.pi, np.pi, 1000)
ejw = np.exp(1j * omega)

X = (8 * np.exp(1j * 2 * omega)) / (1 - 0.5 * np.exp(-1j * omega))

# Амплитуден спектар
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(omega, np.abs(X))
plt.title('Амплитуден спектар |X(e^jω)|')
plt.xlabel('ω')
plt.ylabel('Амплитуда')
plt.grid(True)

# Фазен спектар
plt.subplot(1, 2, 2)
plt.plot(omega, np.angle(X))
plt.title('Фазен спектар ∠X(e^jω)')
plt.xlabel('ω')
plt.ylabel('Фаза (рад)')
plt.grid(True)

plt.tight_layout()
plt.show()
