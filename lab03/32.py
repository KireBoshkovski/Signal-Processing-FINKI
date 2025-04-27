

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# Parameters
frequency = 1  # in Hz
amplitude = 1  # peak value
phase = 0  # in radians
sampling_rate = 100  # in samples per second
duration = 2 * np.pi  # in seconds

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Sine wave
y = amplitude * 0.8 * np.cos(2 * np.pi / 3 * frequency * t + phase) + amplitude * np.cos(5 *
                                                                                         np.pi * frequency * t + phase) * 0.6

# Plotting
plt.plot(t, y)
plt.title('Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
