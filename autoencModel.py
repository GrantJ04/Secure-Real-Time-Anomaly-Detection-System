import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
def createModel(peak, centroid, bandwidth, flatness, rolloff): #creates autoencoder model using given parameters

    x = np.column_stack((peak, centroid, bandwidth, flatness, rolloff)) #forms a matrix
    #each row is one signal sample, each col is a different feature

    scaler = StandardScaler()
    xScaled = scaler.fit_transform(x) #normalizes matrix for use in training

    #split into training and validation sets
    xTrain , xVal = train_test_split(xScaled, test_size = 0.2, random_state = 42) 

    #define autoencoder model

    stackModel = tf.keras.Sequential()

    stackModel.add(layers.Dense(3, activation = 'relu', input_shape=(5,))) #encoder
    stackModel.add(layers.Dense(2, activation = 'relu')) #bottleneck
    stackModel.add(layers.Dense(3, activation = 'relu')) #decoder
    stackModel.add(layers.Dense(5, activation = 'linear')) #output layer reconstruction
    
    stackModel.compile(optimizer = 'adam', loss = 'mse')

    hist = stackModel.fit(xTrain, xTrain, validation_data = (xVal, xVal), epochs = 50, batch_size = 32)

    xValRec = stackModel.predict(xVal) #use trained model to predict each sample

    recErrors = np.mean(np.square(xVal - xValRec), axis = 1) #completes MSE

    threshold = np.percentile(recErrors, 95) #threshold on error distribution

    return stackModel, scaler, threshold

def detectAnomalies(newPeak, newCentroid, newBandwidth, newFlatness, newRolloff, stackModel, scaler, threshold): #detects anomalies on new data

    newData = np.column_stack((newPeak, newCentroid, newBandwidth, newFlatness, newRolloff))

    newScaled = scaler.transform(newData)

    reconstd = stackModel.predict(newScaled)

    recErrors = np.mean(np.square(newScaled - reconstd), axis = 1)

    anomalies = recErrors > threshold
    
    return anomalies, recErrors
