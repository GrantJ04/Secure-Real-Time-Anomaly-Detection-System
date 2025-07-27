import numpy as np
import pandas as pd
import sklearn
import plotly
import flask
import matplotlib.pyplot as plt
from Crypto.Cipher import AES

def set(freqs, fftSignal): #finds/sets specific values from our raw FFT signal for later use
    peak = findPeak(fftSignal)
    centroid = findCentroid(freqs, fftSignal)
    bandwidth = findBandwidth(freqs, fftSignal, centroid)
    flatness = findFlatness(fftSignal)
    rolloff = findRolloff(freqs, fftSignal, .85)
    
def findPeak(signal): #finds peak frequency
    return signal[np.argmax(signal)]

def findCentroid(freqs, signal): #finds spectral centroid
    top = 0
    bottom = 0
    for k in range(len(freqs)):
         #sets numerator
        top += (signal[k] * freqs[k])
        bottom += signal[k]
    if bottom > 0:
        return top/bottom    
    else: 
        return 0

def findBandwidth(freqs, signal, centroid): #finds bandwidth
    top = 0
    bottom = 0
    for k in range(len(freqs)):
        top += signal[k] * np.square(freqs[k] - centroid)
        bottom += signal[k]
    if bottom > 0:
        return np.sqrt(top/bottom)
    else: 
        return 0
    
def findFlatness(signal):
    product = 1
    sSig= 0
    n = 0
    for k in signal:
        product *= k
        sSig += k
        n += 1
    top = np.power(product, 1 / n) #numerator
    bottom = (1/n) * sSig
    if bottom > 0:
        return top/bottom
    else:
        return 0
        
def findRolloff(freqs, signal, percThresh):
    tEnergy = 0
    accum = 0
    for k in range(len(signal)):
        tEnergy += signal[k]

    thresh = percThresh * tEnergy

    for i in range(len(signal)):
        accum += signal[i]
        if accum >= thresh:
            return freqs[i] #returns roll off frequency
    return freqs[-1] #fallback return