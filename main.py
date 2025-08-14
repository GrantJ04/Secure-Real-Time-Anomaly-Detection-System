import numpy as np
import rfGen
import autoencModel
import visualizeResults as viz
from sklearn.model_selection import train_test_split
from autoencModel import choose_threshold_by_f1

# ------------------------------
# 1) Generate synthetic RF data
# ------------------------------
nSignals = 50
times, waves, flags = rfGen.simRFwithPAnomalies(
    nSignals=nSignals,
    anomaly_rate=0.02,          # 2% per sample
    burst_len_range=(20, 40),
    freq_anomaly_prob=0.005     # frequency anomalies
)

# ------------------------------
# 2) Convert signals to features + labels
# ------------------------------
window_size = 128
sample_rate = 1000
allPeaks, allCentroids, allBandwidths, allFlatnesses, allRolloffs, allLabels = rfGen.sigToFeatAndLabels(
    waves, flags, sample_rate, window_size, anomaly_ratio_threshold=0.1
)

# Print dataset stats
print(f"Total windows: {len(allLabels)}")
print(f"Normal windows: {np.sum(allLabels == 0)}")
print(f"Anomalous windows: {np.sum(allLabels == 1)}")
print("Label counts:", dict(zip(*np.unique(allLabels, return_counts=True))))

# ------------------------------
# 3) Split into train/val sets (normal only for AE)
# ------------------------------
idx = np.arange(len(allLabels))
train_idx, val_idx = train_test_split(
    idx, test_size=0.25, random_state=42, stratify=allLabels
)

train_norm_mask = (allLabels[train_idx] == 0)
train_norm_idx = train_idx[train_norm_mask]
if len(train_norm_idx) < 2:
    raise ValueError("Too few normal windows. Increase nSignals or lower anomaly threshold.")

# ------------------------------
# 4) Train autoencoder
# ------------------------------
stackModel, scaler, _ = autoencModel.createModel(
    allPeaks[train_norm_idx],
    allCentroids[train_norm_idx],
    allBandwidths[train_norm_idx],
    allFlatnesses[train_norm_idx],
    allRolloffs[train_norm_idx]
)

# ------------------------------
# 5) Compute validation errors
# ------------------------------
_, val_errors = autoencModel.detectAnomalies(
    allPeaks[val_idx],
    allCentroids[val_idx],
    allBandwidths[val_idx],
    allFlatnesses[val_idx],
    allRolloffs[val_idx],
    stackModel,
    scaler,
    threshold=0.0
)
val_labels = allLabels[val_idx]

# ------------------------------
# 6) Tune threshold using F1
# ------------------------------
threshold = choose_threshold_by_f1(val_errors, val_labels)
print(f"Chosen threshold: {threshold:.6f}")

# ------------------------------
# 7) Apply threshold to all data
# ------------------------------
pred_all, recErrors = autoencModel.detectAnomalies(
    allPeaks,
    allCentroids,
    allBandwidths,
    allFlatnesses,
    allRolloffs,
    stackModel,
    scaler,
    threshold
)

# ------------------------------
# 8) Print classification summary
# ------------------------------
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
pred_labels = pred_all.astype(int)
cm = confusion_matrix(allLabels, pred_labels)
print("\n=== Confusion Matrix ===")
print(cm)
print("TN =", cm[0,0], "FP =", cm[0,1], "FN =", cm[1,0], "TP =", cm[1,1])
print("\n=== Classification Report ===")
print(classification_report(allLabels, pred_labels, target_names=["Normal","Anomaly"]))
print("Precision:", precision_score(allLabels, pred_labels, zero_division=0))
print("Recall:", recall_score(allLabels, pred_labels, zero_division=0))
print("F1 Score:", f1_score(allLabels, pred_labels, zero_division=0))

# ------------------------------
# 9) Visualize
# ------------------------------
viz.plotRecErrorDist(recErrors, threshold)
viz.plotErrorOverTime(recErrors, allLabels, pred_all, threshold)
viz.plotConfusionMatrix(allLabels, pred_all)
viz.plotPrecisionRecallCurve(allLabels, recErrors)
