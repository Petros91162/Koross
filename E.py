import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
f1 = 50  # Frequency 1 in Hz
f2 = 120  # Frequency 2 in Hz
fs = 1000  # Sampling frequency in Hz
t = np.arange(0, 1, 1/fs)  # Time vector for 1 second

# Define the signal
s = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# Compute the FFT
S = np.fft.fft(s)
freq = np.fft.fftfreq(len(s), 1/fs)

# Plot the signal
plt.figure(figsize=(12, 6))

# Time domain plot
plt.subplot(2, 1, 1)
plt.plot(t, s)
plt.title('Signal in Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Frequency domain plot
plt.subplot(2, 1, 2)
plt.plot(freq[:len(freq)//2], np.abs(S)[:len(S)//2])  # Plot only the positive frequencies
plt.title('Signal in Frequency Domain')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
