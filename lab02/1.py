import matplotlib

import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

# 1.
# Со користење на функцијата за цртање график во Пајтон, да се прикаже на график функцијата
# f(t)=sin(0,7πt+0,4), во опсегот (0,10). За графикот да се употребат 1000 точки.

t = np.linspace(0, 10, 1000)
print('t: ', t)
func = np.sin(0.7 * np.pi * t + 0.4)
plt.plot(t, func)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()

# 2.
# Со користење на функцијата за цртање стапчиња да се прикаже низата x[n]=sin(0,7πn+0,4),
# за времиња n=0,1,…10. Низата да се прикаже на ист цртеж со графикот на функцијата во претходната задача

n = list(range(0, 11))
nn = np.linspace(0, 10, num=11,dtype=int)
print('nn:', nn)
x = np.sin(0.7 * np.pi * nn + 0.4)
# x = np.zeros(len(n))
# for i in range(len(n)):
#     x[i] = np.sin(0.7 * np.pi * n[i] + 0.4)
plt.plot(t, func)
plt.stem(n, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()

# 3.
# Да се земат примероци од функцијата на растојание Ts=0,125s и да се прикажат на ист цртеж со функцијата

# Параметри
omega = 2 * np.pi * 0.25  # 0.25 Hz
phi = 0
Ts = 0.125
omega_0 = omega * Ts


def f(t):
    return np.sin(omega * t + phi)


def x(n):
    return np.sin(omega_0 * n + phi)


# Временска оска за примероците
n_samples = np.arange(0 / Ts, 10 / Ts)  # Индекси на примероците
t_samples = n_samples * Ts  # Временски точки за примероците

y_continuous = f(t)
y_samples = x(n_samples)

plt.plot(t, y_continuous)
plt.stem(t_samples, y_samples)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()
