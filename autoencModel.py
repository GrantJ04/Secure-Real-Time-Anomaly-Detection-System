import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score

def createModel(peak, centroid, bandwidth, flatness, rolloff, labels=None, bottleneck_size=3):
    """
    Creates an autoencoder for anomaly detection.
    
    Parameters:
    - peak, centroid, bandwidth, flatness, rolloff: feature arrays
    - labels: array of 0 (normal) / 1 (anomaly), optional (used for F1-optimal threshold)
    - bottleneck_size: integer size of bottleneck layer
    """
    
    # Stack features into matrix
    X = np.column_stack((peak, centroid, bandwidth, flatness, rolloff))
    
    # Only use normal samples for training
    if labels is not None:
        normal_mask = (labels == 0)
        X_normal = X[normal_mask]
    else:
        X_normal = X
    
    # Scale
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    # Train/validation split (normal data only)
    X_train, X_val = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
    
    # Build autoencoder
    model = tf.keras.Sequential([
        layers.Dense(4, activation='relu', input_shape=(5,)),  # encoder
        layers.Dense(bottleneck_size, activation='relu'),      # bottleneck
        layers.Dense(4, activation='relu'),                    # decoder
        layers.Dense(5, activation='linear')                   # reconstruction
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=50, batch_size=32, verbose=1)
    
    # Compute reconstruction error on validation set
    X_val_pred = model.predict(X_val)
    val_errors = np.mean(np.square(X_val - X_val_pred), axis=1)
    
    # Choose threshold
    if labels is None:
        threshold = np.percentile(val_errors, 95)
    else:
        # Use F1-optimal threshold if labels provided
        val_labels = labels[normal_mask][len(X_train):]  # align val labels
        prec, rec, thresh = precision_recall_curve(val_labels, val_errors)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_scores)
        threshold = thresh[best_idx]
        print(f"Best threshold by F1: {threshold:.4f} (F1={f1_scores[best_idx]:.4f})")
    
    return model, scaler, threshold


def detectAnomalies(newPeak, newCentroid, newBandwidth, newFlatness, newRolloff, model, scaler, threshold):
    """
    Detects anomalies using trained autoencoder model.
    """
    new_data = np.column_stack((newPeak, newCentroid, newBandwidth, newFlatness, newRolloff))
    new_scaled = scaler.transform(new_data)
    
    reconstructed = model.predict(new_scaled)
    rec_errors = np.mean(np.square(new_scaled - reconstructed), axis=1)
    
    anomalies = rec_errors > threshold
    return anomalies, rec_errors
