import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import plotly
import flask
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
import time

def datastream(): #loads datastream
    file = pd.read_csv("creditcard.csv")
    
    prevTime = 0
    for _, row in file.iterrows():
        amt = row['Amount']
        cls = row['Class']
        if prevTime == 0:
            delay = 0
        else:
            delay = row['Time'] - prevTime
            if delay < 0:
                delay = 0
        prevTime = row['Time']
        
        print(f"Amount: ${amt:<10} | Class: {cls:<5} | Delay {delay:<6.2f}")
        time.sleep(delay)

def trainSave():
    data = pd.read_csv("creditcard.csv")
    x = data.drop('Class', axis = 1) #all cols except class
    y = data['Class'] #class col

    #split into training and testing sets

    #80train/20test
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler() #creates a scalar

    xtrainScaled = scaler.fit_transform(xtrain)
    xtestScaled = scaler.transform(xtest)

    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(xtrainScaled, ytrain)

    yPred = model.predict(xtestScaled)
    print(classification_report(ytest, yPred))
    joblib.dump(model, 'fraud_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved!")

if __name__ == "__main__":
    #datastream()
    trainSave()