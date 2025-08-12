import seaborn as sns
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt

def plotRecErrorDist(errors, threshold):
    sns.kdeplot(errors)
    plt.axvline(threshold, color = 'r', linestyle = '--')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("Reconstruction Error Distribution with Threshold")
    plt.show()

def plotErrorOverTime(errors, trueLabels, predictedLabels, threshold):
    inds = np.arange(len(errors))

    plt.figure(figsize = (10,6))
    sns.lineplot(x=inds, y=errors, label = 'Reconstruction Error')
    plt.axhline(y=threshold, color='r', linestyle = '--', label = 'Threshold')

    anomInds = inds[predictedLabels == 1]
    anomErrors = errors[predictedLabels == 1]
    sns.scatterplot(x=anomInds, y=anomErrors, color = 'red', label = 'Predicted Anomaly', s=50)

    plt.xlabel("Window Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error Over Time with Anomalies")
    plt.legend()
    plt.show()

def plotConfusionMatrix(trueLabels, predictedLabels):
    confMatrix = skm.confusion_matrix(trueLabels, predictedLabels)
    plt.figure(figsize = (6,5))
    sns.heatmap(confMatrix, annot = True, fmt = 'd', cmap = 'Blues', cbar = False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(ticks = [0.5,1.5], labels = ['Normal', 'Anomaly'])
    plt.yticks(ticks = [0.5,1.5], labels = ['Normal', 'Anomaly'], rotation = 0)
    plt.show()

def plotPrecisionRecallCurve(trueLabels, scores):
    precision, recall, _ = skm.precision_recall_curve(trueLabels, scores)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()