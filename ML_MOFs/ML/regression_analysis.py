import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import r2_score

regression_targets = ["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "SC CO2 loading (mol/kg)",
                      "SC CH4 loading (mol/kg)", "TSN", "LOG10 TSN"]
methods = ["MLR", "RF", "SVM"]

# histogram of error vs target range
for method in methods:
    data = pd.read_csv("..\\Results\\ML_results\\regression\\" + method + "_predictions.csv")
    for target in regression_targets:
        data[target + " Error"] = np.array(data[target + " Prediction"]) - np.array(data[target])
        fig = px.histogram(data, x=target + " Error", width=600,
                           height=400, title=target + " Errors - " + method)
        target = target.replace("(", "_")
        target = target.replace(")", "_")
        target = target.replace("/", "_")
        filename = "../Graphs/ML_graphs/Regression/" + method + "_" + target + "_error_histogram.png"
        fig.write_image(filename, scale=2)


# scatter of error vs target
for method in methods:
    data = pd.read_csv("..\\Results\\ML_results\\regression\\" + method + "_predictions.csv")
    for target in regression_targets:
        data[target + " Error"] = np.array(data[target + " Prediction"]) - np.array(data[target])
        fig = px.scatter(data, x=target, y=target + " Error", width=600,
                         height=400, title=target + " Error vs " + target + " - " + method)
        target = target.replace("(", "_")
        target = target.replace(")", "_")
        target = target.replace("/", "_")
        filename = "../Graphs/ML_graphs/Regression/" + method + "_" + target + "_error_scatter.png"
        fig.write_image(filename, scale=2)


df = pd.read_csv("..\\Results\\ML_results\\regression\\RF_importance.csv")
# feature importance for each target
data = [go.Bar(name=x, x=df["Descriptor"], y=df[x + " Mean"]) for x in regression_targets]
fig = go.Figure(data=data)
for i in range(len(regression_targets)):
    colors = []
    for j in range(len(df["Descriptor"])):
        colors.append(px.colors.qualitative.D3[i])
    fig.data[i].marker.color = tuple(colors)
# Change the bar mode
fig.update_layout(barmode='group', width=1000, height=600, title="Feature Importance - RF")
fig.update_xaxes(tickangle=90)
filename = "../Graphs/ML_graphs/Regression/RF_importance.png"
fig.write_image(filename, scale=2)
