import numpy as np
import rfGen
import fftSignal
import setValsForUse
import autoencModel
def main():
    nSamples = 50
    windowSize = 100  # choose suitable window size
    sampleRate = 1000 # same as used in rfGen

    allPeaks, allCentroids, allBandwidths, allFlatnesses, allRolloffs, allLabels = [], [], [], [], [], []

    for _ in range(nSamples):
        points, spikedWave , anomalyFlags = rfGen.simRFwithPAnomalies(5)

        peaks, centroids, bandwidths, flatnesses, rolloffs, labels = rfGen.sigToFeatAndLabels(
            spikedWave,
            anomalyFlags,
            sampleRate,
            windowSize,
            anomaly_ratio_threshold=0.3
        )

        allPeaks.extend(peaks)
        allCentroids.extend(centroids)
        allBandwidths.extend(bandwidths)
        allFlatnesses.extend(flatnesses)
        allRolloffs.extend(rolloffs)
        allLabels.extend(labels)
    
    allPeaks = np.array(allPeaks) #conversions to numpy arrays
    allCentroids = np.array(allCentroids)
    allBandwidths = np.array(allBandwidths)
    allFlatnesses = np.array(allFlatnesses)
    allRolloffs = np.array(allRolloffs)
    allLabels = np.array(allLabels)
    
    normalIndices = np.where(allLabels == 0)[0] #train on only normal samples
    stackModel, scaler, threshold = autoencModel.createModel(
        allPeaks[normalIndices],
        allCentroids[normalIndices],
        allBandwidths[normalIndices],
        allFlatnesses[normalIndices],
        allRolloffs[normalIndices]
    )

    anomalies, recErrors = autoencModel.detectAnomalies(
        allPeaks,
        allCentroids,
        allBandwidths,
        allFlatnesses,
        allRolloffs,
        stackModel,
        scaler,
        threshold
    )

    #print results to user
    for k, isAnom in enumerate(anomalies):
        true_label = "ANOMALOUS" if allLabels[k] == 1 else "normal"
        predicted = "ANOMALOUS" if isAnom else "normal"
        print(f"Window {k}: True: {true_label}, Predicted: {predicted}, Reconstruction error = {recErrors[k]:.4f}")

if __name__ == "__main__":
    main()
