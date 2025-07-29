import numpy as np
import pandas as pd
import sklearn
import plotly
import flask
import matplotlib.pyplot as plt
from Crypto.Cipher import AES

def setVals(freqs, fftSignal): #finds/sets specific values from our raw FFT signal for later use
    peak = findPeak(fftSignal)
    centroid = findCentroid(freqs, fftSignal)
    bandwidth = findBandwidth(freqs, fftSignal, centroid)
    flatness = findFlatness(fftSignal)
    rolloff = findRolloff(freqs, fftSignal, .85)
    return peak, centroid, bandwidth, flatness, rolloff
    
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
    signal = np.array(signal)
    signal = signal[signal > 0]  # avoid log(0) and negatives

    if len(signal) == 0:
        return 0  # fallback if all bins were zero or negative

    log_geom_mean = np.exp(np.mean(np.log(signal)))  # geometric mean
    arith_mean = np.mean(signal)  # arithmetic mean

    return log_geom_mean / arith_mean if arith_mean > 0 else 0

        
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