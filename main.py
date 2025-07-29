import numpy as np
import pandas as pd
import sklearn
import plotly
import flask
import matplotlib.pyplot as plt
from Crypto.Cipher import AES

import rfGen #imports rf file for use
import fftSignal #imports fftSignal file for use
import setValsForUse #imports setValsForUse file for use
import autoencModel #imports autoencModel file for use

def main():
    nSamples = 50
    allWaves = []
    allPoints = None
    for _ in range(nSamples):
        points, spikedWave = rfGen.simRFwithPAnomalies(5)
        allWaves.append(spikedWave)
        if allPoints is None:
            allPoints = points #simply sets points once since they are always the same
    
    allPeaks = [] #sets empty lists for each important value
    allCentroids = []
    allBandwidths = []
    allFlatnesses = []
    allRolloffs = []
    for wave in allWaves:
        fftResult, freqs = fftSignal.transform(allPoints, wave)
        peak, centroid, bandwidth, flatness, rolloff = setValsForUse.setVals(freqs, np.abs(fftResult))
        allPeaks.append(peak)
        allCentroids.append(centroid)
        allBandwidths.append(bandwidth)
        allFlatnesses.append(flatness)
        allRolloffs.append(rolloff)

    allPeaks = np.array(allPeaks) #conversions to numpy arrays
    allCentroids = np.array(allCentroids)
    allBandwidths = np.array(allBandwidths)
    allFlatnesses = np.array(allFlatnesses)
    allRolloffs = np.array(allRolloffs)
    
    stackModel, scaler, threshold = autoencModel.createModel(allPeaks, allCentroids, allBandwidths, allFlatnesses, allRolloffs)

    anomalies, recErrors = autoencModel.detectAnomalies(allPeaks, allCentroids,allBandwidths,allFlatnesses,allRolloffs, stackModel, scaler, threshold)

    #print results to user
    for k, isAnom in enumerate(anomalies):
        status = "ANOMALOUS" if isAnom else "normal"
        print(f"Sample {k}: {status}, Reconstruction error = {recErrors[k]:.4f}")

if __name__ == "__main__":
    main()
