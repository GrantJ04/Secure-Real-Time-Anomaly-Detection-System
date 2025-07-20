import numpy as np
import pandas as pd
import sklearn
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

if __name__ == "__main__":
    datastream()