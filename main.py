import numpy as np
import rfGen
import autoencModel
import visualizeResults as viz
import dashboard as db
from sklearn.model_selection import train_test_split
from autoencModel import choose_threshold_by_f1
from Crypto.Random import get_random_bytes
from cryptoUtils import decryptData, encryptData
import pandas as pd
import joblib
import kaggle  # your financial model file
import os
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
    anomaly_rate=0.02,
    burst_len_range=(20, 40),
    freq_anomaly_prob=0.005
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

# ------------------------------
# 3) Split into train/val sets (normal only for AE)
# ------------------------------
idx = np.arange(len(allLabels))
train_idx, val_idx = train_test_split(
    idx, test_size=0.25, random_state=42, stratify=allLabels
)
train_norm_mask = (allLabels[train_idx] == 0)
train_norm_idx = train_idx[train_norm_mask]

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
# 7a) Secure predictions
# ------------------------------
predData = {
    "predLabels": predAll.astype(int).tolist(),
    "recErrors": recErrors.tolist()
}
predStr = json.dumps(predData)
encryptedPreds = encryptData(predStr, aesKey)
decryptedPreds = json.loads(decryptData(encryptedPreds, aesKey))
predLabels = np.array(decryptedPreds["predLabels"])
recErrors = np.array(decryptedPreds["recErrors"])

# ------------------------------
# 8) Financial model: train/load and prepare data
# ------------------------------
# Only train if missing
if not os.path.exists("fraud_model.pkl") or not os.path.exists("scaler.pkl"):
    kaggle.trainSave()

scaler_fin = joblib.load("scaler.pkl")
model_fin = joblib.load("fraud_model.pkl")

# Precompute predictions if missing
if os.path.exists("finance_preds.csv"):
    df_fin = pd.read_csv("finance_preds.csv")
    y_fin = df_fin['y_true'].values
    y_pred_fin = df_fin['y_pred'].values
else:
    finance_data = pd.read_csv("creditcard.csv")
    x_fin = finance_data.drop("Class", axis=1)
    y_fin = finance_data["Class"].values

    x_scaled = scaler_fin.transform(x_fin)
    y_pred_fin = model_fin.predict(x_scaled)

    pd.DataFrame({'y_true': y_fin, 'y_pred': y_pred_fin}).to_csv("finance_preds.csv", index=False)

# ------------------------------
# 9) Launch dashboard
# ------------------------------
db.db(recErrors, threshold, allLabels, predAll, recErrors, finance_pred_file="finance_preds.csv")
