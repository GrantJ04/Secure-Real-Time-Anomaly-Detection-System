import numpy as np
import pandas as pd
import sklearn
import plotly
import flask
import matplotlib.pyplot as plt
from Crypto.Cipher import AES

import rf #imports rf file for use

def main():
    points, waves = rf.simRFwithPAnomalies(5) 
    rf.plotWaves(points, waves)



if __name__ == "__main__":
    main()