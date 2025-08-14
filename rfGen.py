import numpy as np
import matplotlib.pyplot as plt
import setValsForUse

def plotWaves(points, waves):
    plt.plot(points, waves)
    plt.title("Simulated RF Signal")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.show()

def simRFwithPAnomalies(
    nSignals,
    anomaly_rate=0.02,       # probability of anomaly event per sample
    burst=True,              # whether to allow bursts
    burst_len_range=(8, 25), # burst length in samples
    freq_anomaly_prob=0.005, # chance of frequency anomaly
    sample_rate=1000         # samples per second
):
    """
    Simulates an RF signal with optional spike bursts and frequency anomalies.
    Returns:
        times (array)         - time points in seconds
        spikedWaves (array)   - simulated RF amplitudes
        anomalyFlags (array)  - True if the point is anomalous
    """

    # Simulate 2π seconds worth of data at given resolution
    points = np.arange(0, 2 * np.pi, 0.01)
    times = np.arange(len(points)) / sample_rate

    # Random base waveform parameters
    randWF = [np.random.uniform(0.1, 4) for _ in range(nSignals)]   # frequencies
    randAMP = [np.random.uniform(0.5, 2) for _ in range(nSignals)]  # amplitudes
    randPHS = [np.random.uniform(0, 2 * np.pi) for _ in range(nSignals)] # phases

    # State variables
    spikedWaves = []
    anomalyFlags = []
    driftACC = [0] * nSignals
    in_burst = 0  # counts down burst samples

    for idx, point in enumerate(points):
        noise = np.random.normal(-0.07, 0.07)
        signal = 0
        isAnom = False

        # Small drift in frequency for realism
        for drift in range(nSignals):
            driftACC[drift] += np.random.uniform(-0.001, 0.001)

        # Check if in an anomaly burst
        if in_burst > 0:
            spike = np.random.uniform(0.5, 1.0)
            if np.random.rand() < 0.5:
                spike = -spike
            signal += spike
            isAnom = True
            in_burst -= 1

        # Randomly trigger a burst
        elif burst and np.random.rand() < anomaly_rate:
            in_burst = np.random.randint(burst_len_range[0], burst_len_range[1])

        # Frequency anomaly (sudden frequency jump)
        if np.random.rand() < freq_anomaly_prob:
            target = np.random.randint(0, nSignals)
            randWF[target] += np.random.uniform(-1.5, 1.5)
            isAnom = True

        # Build the composite waveform
        for wave in range(nSignals):
            signal += randAMP[wave] * np.sin(
                (driftACC[wave] + randWF[wave]) * point + randPHS[wave]
            )

        signal += noise
        spikedWaves.append(signal)
        anomalyFlags.append(isAnom)

    return times, np.array(spikedWaves), np.array(anomalyFlags)

def sigToFeatAndLabels(signal, anomFlags, sampleRate, windowSize, anomaly_ratio_threshold=0.05):
    nWindows = len(signal) // windowSize
    peaks, centroids, bandwidths, flatnesses, rolloffs = [], [], [], [], []
    labels = []

    for k in range(nWindows):
        startP = k * windowSize
        endP = startP + windowSize
        window = signal[startP:endP]
        windowFlags = anomFlags[startP:endP]

        fftVals = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(len(window), d=1/sampleRate)

        peak, centroid, bandwidth, flatness, rolloff = setValsForUse.setVals(freqs, fftVals)

        peaks.append(peak)
        centroids.append(centroid)
        bandwidths.append(bandwidth)
        flatnesses.append(flatness)
        rolloffs.append(rolloff)

        # Label window as anomalous if fraction of flagged points exceeds threshold
        anomaly_ratio = np.mean(windowFlags)
        labels.append(int(anomaly_ratio > anomaly_ratio_threshold))

    return (
        np.array(peaks),
        np.array(centroids),
        np.array(bandwidths),
        np.array(flatnesses),
        np.array(rolloffs),
        np.array(labels),
    )

        
# Example usage
if __name__ == "__main__":
    times, waves, flags = simRFwithPAnomalies(
        nSignals=3,
        anomaly_rate=0.005,
        burst=True,
        freq_anomaly_prob=0.002
    )

    plotWaves(times, waves)

    # Optional: visualize anomalies in red
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(times, waves, label="RF Signal")
    plt.scatter(times[flags], waves[flags], color='red', label="Anomalies")
    plt.legend()
    plt.show()
