import numpy as np
import pandas as pd
import sklearn
import plotly
import flask
import matplotlib.pyplot as plt
from Crypto.Cipher import AES

import rfGen #imports rf file for use
import fftSignal #imports fftSignal file for us

def main():
    points, spikedWaves = rfGen.simRFwithPAnomalies(5) 
    rfGen.plotWaves(points, spikedWaves)
    fftResult, freqs = fftSignal.transform(points,spikedWaves)
    fftSignal.plotFFT(freqs, np.abs(fftResult))

if __name__ == "__main__":
    main()
