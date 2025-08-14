import numpy as np

def simRFwithPAnomalies(
    nSignals=50,
    anomaly_rate=0.02,
    burst_len_range=(10,40),
    freq_anomaly_prob=0.005,
    nPoints=1000,
    debug=False
):
    """
    Simulate RF signals with anomalies.

    Returns:
    - times: list of time arrays
    - waves: list of wave arrays
    - flags: list of boolean arrays indicating anomalies
    """
    times = []
    waves = []
    flags = []

    for _ in range(nSignals):
        t = np.linspace(0, 2*np.pi, nPoints)
        base_wave = np.sin(t)
        anomaly_flag = np.zeros_like(base_wave, dtype=int)

        i = 0
        while i < nPoints:
            if np.random.rand() < anomaly_rate:
                burst_len = np.random.randint(burst_len_range[0], burst_len_range[1])
                end_idx = min(i + burst_len, nPoints)
                base_wave[i:end_idx] += np.random.uniform(0.5, 1.5, end_idx-i)
                anomaly_flag[i:end_idx] = 1
                i = end_idx
            else:
                i += 1
        
        # Ensure at least a few normal samples for small nSignals
        if debug and np.sum(anomaly_flag==0) < nPoints*0.2:
            normal_indices = np.random.choice(nPoints, size=int(nPoints*0.2), replace=False)
            anomaly_flag[normal_indices] = 0
        
        times.append(t)
        waves.append(base_wave)
        flags.append(anomaly_flag)

    return times, waves, flags


def sigToFeatAndLabels(
    waves, 
    flags, 
    sample_rate, 
    window_size=128, 
    anomaly_ratio_threshold=0.1
):
    """
    Convert signals into windowed features and labels.
    Returns:
    - peaks, centroids, bandwidths, flatness, rolloffs, labels
    """
    allPeaks, allCentroids, allBandwidths, allFlatnesses, allRolloffs, allLabels = [], [], [], [], [], []

    for wave, flag in zip(waves, flags):
        nWindows = len(wave) // window_size
        for i in range(nWindows):
            start = i*window_size
            end = start+window_size
            w = wave[start:end]
            f = flag[start:end]

            # Features
            peak = np.max(np.abs(w))
            centroid = np.sum(np.arange(window_size)*np.abs(w))/np.sum(np.abs(w))
            bandwidth = np.std(w)
            flatness = np.exp(np.mean(np.log(np.abs(w)+1e-8))) / (np.mean(np.abs(w))+1e-8)
            rolloff = np.percentile(np.abs(w), 85)

            allPeaks.append(peak)
            allCentroids.append(centroid)
            allBandwidths.append(bandwidth)
            allFlatnesses.append(flatness)
            allRolloffs.append(rolloff)

            # Label
            anomaly_ratio = np.sum(f)/window_size
            allLabels.append(int(anomaly_ratio > anomaly_ratio_threshold))

    return (
        np.array(allPeaks),
        np.array(allCentroids),
        np.array(allBandwidths),
        np.array(allFlatnesses),
        np.array(allRolloffs),
        np.array(allLabels)
    )
