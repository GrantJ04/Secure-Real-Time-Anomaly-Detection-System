import numpy as np
import rfGen
import autoencModel
import visualizeResults as viz
import dashboard as db
from sklearn.model_selection import train_test_split
from autoencModel import choose_threshold_by_f1
from Crypto.Random import get_random_bytes
from cryptoUtils import decryptData, encryptData

import json

# ------------------------------
# 0) Generate AES key
# ------------------------------
aesKey = get_random_bytes(16)

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

# ------------------------------
# 2a) Secure features: encrypt
# ------------------------------
featureData = {
    "peaks": allPeaks.tolist(),
    "centroids": allCentroids.tolist(),
    "bandwidths": allBandwidths.tolist(),
    "flatnesses": allFlatnesses.tolist(),
    "rolloffs": allRolloffs.tolist(),
    "labels": allLabels.tolist()
}
featureStr = json.dumps(featureData)
encryptedFeats = encryptData(featureStr, aesKey)

# ------------------------------
# 2b) Decrypt before using in model
# ------------------------------
decryptedFeats = decryptData(encryptedFeats, aesKey)
decryptedData = json.loads(decryptedFeats)
allPeaks = np.array(decryptedData["peaks"])
allCentroids = np.array(decryptedData["centroids"])
allBandwidths = np.array(decryptedData["bandwidths"])
allFlatnesses = np.array(decryptedData["flatnesses"])
allRolloffs = np.array(decryptedData["rolloffs"])
allLabels = np.array(decryptedData["labels"])

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
predAll, recErrors = autoencModel.detectAnomalies(
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
# 7a) Optionally secure predictions
# ------------------------------
predData = {
    "predLabels": predAll.astype(int).tolist(),
    "recErrors": recErrors.tolist()
}
predStr = json.dumps(predData)
encryptedPreds = encryptData(predStr, aesKey)

# Decrypt before visualization
decryptedPreds = json.loads(decryptData(encryptedPreds, aesKey))
predLabels = np.array(decryptedPreds["predLabels"])
recErrors = np.array(decryptedPreds["recErrors"])

# ------------------------------
# 8) Print classification summary
# ------------------------------
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
cm = confusion_matrix(allLabels, predLabels)
print("\n=== Confusion Matrix ===")
print(cm)
print("TN =", cm[0,0], "FP =", cm[0,1], "FN =", cm[1,0], "TP =", cm[1,1])
print("\n=== Classification Report ===")
print(classification_report(allLabels, predLabels, target_names=["Normal","Anomaly"]))
print("Precision:", precision_score(allLabels, predLabels, zero_division=0))
print("Recall:", recall_score(allLabels, predLabels, zero_division=0))
print("F1 Score:", f1_score(allLabels, predLabels, zero_division=0))

# ------------------------------
# 9) Visualize using Dash
# ------------------------------
db.db(recErrors, threshold, allLabels, predAll, recErrors)

