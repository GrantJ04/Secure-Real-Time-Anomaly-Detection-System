import numpy as np
import matplotlib.pyplot as plt

def plotWaves(points, waves):
    plt.plot(points, waves)
    plt.title("Simulated RF Signal")
    plt.xlabel("Radians")
    plt.ylabel("Amplitude")
    positions = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    plt.xticks(positions, labels)
    plt.show()

def simRFwithPAnomalies(nSignals):  # simulates RF signal with spike anomalies
    points = np.arange(0, 2 * np.pi, 0.01)
    randWF = []
    randAMP = []
    randPHS = []

    for _ in range(nSignals):
        randWF.append(np.random.uniform(0.1, 4))
        randAMP.append(np.random.uniform(0.5, 2))
        randPHS.append(np.random.uniform(0, 2 * np.pi))

    spikedWaves = []
    anomalyFlags = []

    driftACC = [0] * nSignals

    for point in points:
        noise = np.random.normal(-0.07, 0.07)
        signal = 0
        isAnom = False  # ← FIX: set default for each sample

        for drift in range(nSignals):
            driftACC[drift] += np.random.uniform(-0.001, 0.001)

        randS = np.random.uniform(0, 1)
        if randS < 0.01:  # 1% chance for a spike
            spike = np.random.uniform(0.3, 0.8)
            if np.random.rand() < 0.5:
                spike = -spike
            signal += spike
            isAnom = True  # mark this point as anomalous

        for wave in range(nSignals):
            signal += randAMP[wave] * np.sin((driftACC[wave] + randWF[wave]) * point + randPHS[wave])

        signal += noise
        spikedWaves.append(signal)
        anomalyFlags.append(isAnom)

    return points, spikedWaves, anomalyFlags
