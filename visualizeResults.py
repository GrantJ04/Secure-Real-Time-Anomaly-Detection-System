# viz.py
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve
import numpy as np

def plotRecErrorDist(errors, threshold):
    """
    Plot reconstruction error distribution as a simple line plot with threshold.
    """
    errors = np.array(errors)
    sorted_errors = np.sort(errors)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(sorted_errors)),
        y=sorted_errors,
        mode='lines',
        name='Reconstruction Error',
        line=dict(color='blue')
    ))
    
    # Add threshold line
    fig.add_hline(y=threshold, line=dict(color='red', dash='dash'), annotation_text='Threshold')
    
    fig.update_layout(
        title="Reconstruction Error Distribution with Threshold",
        xaxis_title="Index (sorted errors)",
        yaxis_title="Reconstruction Error"
    )
    
    return fig

def plotErrorOverTime(errors, trueLabels, predictedLabels, threshold):
    """
    Plot reconstruction error over time with predicted anomalies highlighted.
    """
    inds = list(range(len(errors)))
    fig = go.Figure()
    
    # Error line
    fig.add_trace(go.Scatter(
        x=inds,
        y=errors,
        mode='lines',
        name='Reconstruction Error'
    ))
    
    # Threshold line
    fig.add_hline(y=threshold, line=dict(color='red', dash='dash'), annotation_text='Threshold')
    
    # Predicted anomalies
    anomInds = [i for i, p in enumerate(predictedLabels) if p == 1]
    anomErrors = [errors[i] for i in anomInds]
    fig.add_trace(go.Scatter(
        x=anomInds,
        y=anomErrors,
        mode='markers',
        marker=dict(color='red', size=10),
        name='Predicted Anomaly'
    ))
    
    fig.update_layout(
        title="Reconstruction Error Over Time with Anomalies",
        xaxis_title="Window Index",
        yaxis_title="Reconstruction Error"
    )
    return fig

def plotConfusionMatrix(trueLabels, predictedLabels):
    """
    Plot confusion matrix as annotated heatmap.
    """
    confMatrix = confusion_matrix(trueLabels, predictedLabels)
    fig = ff.create_annotated_heatmap(
        z=confMatrix,
        x=['Normal', 'Anomaly'],
        y=['Normal', 'Anomaly'],
        colorscale='Blues',
        showscale=True
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    return fig

def plotPrecisionRecallCurve(trueLabels, scores):
    """
    Plot precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(trueLabels, scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines+markers',
        name='Precision-Recall Curve',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0,1]),
        yaxis=dict(range=[0,1])
    )
    return fig

def plotFinancialCounts(y_true, y_pred, n_buckets=300):
    """
    Aggregate financial predictions into buckets and plot counts.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    bucket_size = max(1, len(y_true)//n_buckets)
    buckets = np.arange(0, len(y_true), bucket_size)
    
    true_counts = [y_true[i:i+bucket_size].sum() for i in buckets]
    pred_counts = [y_pred[i:i+bucket_size].sum() for i in buckets]

    df = {
        'Bucket': list(range(len(buckets))),
        'True': true_counts,
        'Predicted': pred_counts
    }

    fig = px.bar(df, x='Bucket', y=['True', 'Predicted'],
                 barmode='group',
                 labels={'value':'Number of Fraud Cases', 'Bucket':'Data Bucket'},
                 color_discrete_map={'True':'royalblue','Predicted':'tomato'})

    fig.update_layout(template='plotly_dark', title='Fraud Counts per Bucket', font=dict(color='white'))
    return fig