import numpy as np
import matplotlib.pyplot as plt
def transform(points, waves): #function for Fast-Fourier Transform
    result = np.fft.fft(waves) #fft of waves
    samplePoint = points[1] - points[0] #pulls a sample interval
    freqs = np.fft.fftfreq(len(waves), d = samplePoint) #freq bins of fft
    return result, freqs

def plotFFT(freq, fftMag):
    half = len(freq) // 2
    dbMag = 20 * np.log10(fftMag[:half] + 1e-10)
    plt.plot(freq[:half], dbMag)
    plt.xlabel("Frequency (rad/sample)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.title("FFT Magnitude Spectrum")
    plt.xlim(0, 10)
    plt.show()
