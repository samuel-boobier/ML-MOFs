import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import r2_score

regression_targets = ["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "SC CO2 loading (mol/kg)",
                      "SC CH4 loading (mol/kg)", "TSN", "LOG10 TSN"]
# methods = ["MLR", "RF", "SVM"]
#
# # histogram of error vs target range
# for method in methods:
#     data = pd.read_csv("..\\Results\\ML_results\\regression\\" + method + "_predictions.csv")
#     for target in regression_targets:
#         data[target + " Error"] = np.array(data[target + " Prediction"]) - np.array(data[target])
#         fig = px.histogram(data, x=target + " Error", width=600,
#                            height=400, title=target + " Errors - " + method)
#         target = target.replace("(", "_")
#         target = target.replace(")", "_")
#         target = target.replace("/", "_")
#         filename = "../Graphs/ML_graphs/Regression/" + method + "_" + target + "_error_histogram.png"
#         fig.write_image(filename, scale=2)
#
#
# # scatter of error vs target
# for method in methods:
#     data = pd.read_csv("..\\Results\\ML_results\\regression\\" + method + "_predictions.csv")
#     for target in regression_targets:
#         data[target + " Error"] = np.array(data[target + " Prediction"]) - np.array(data[target])
#         fig = px.scatter(data, x=target, y=target + " Error", width=600,
#                          height=400, title=target + " Error vs " + target + " - " + method)
#         target = target.replace("(", "_")
#         target = target.replace(")", "_")
#         target = target.replace("/", "_")
#         filename = "../Graphs/ML_graphs/Regression/" + method + "_" + target + "_error_scatter.png"
#         fig.write_image(filename, scale=2)

axis_titles = {
    "CO2 loading (mol/kg)": "BM CO<sub>2</sub> Loading",
    "CH4 loading (mol/kg)": "BM CH<sub>4</sub> Loading",
    "SC CO2 loading (mol/kg)": "SC CO<sub>2</sub> Loading",
    "SC CH4 loading (mol/kg)": "SC CH<sub>4</sub> Loading",
    "TSN": "TSN",
    "LOG10 TSN": "log<sub>10</sub> TSN"
}

df = pd.read_csv("..\\Results\\ML_results\\regression\\RF_importance.csv")
# feature importance for each target
data = [go.Bar(name=axis_titles[x], x=df["Descriptor"], y=df[x + " Mean"], error_y=dict(type='data',
                                                                                        thickness=1,
                                                                                        width=2,
                                                                                        array=df[x + " SD"])) for x in
        regression_targets]
fig = go.Figure(data=data)
for i in range(len(regression_targets)):
    colors = []
    for j in range(len(df["Descriptor"])):
        colors.append(px.colors.qualitative.D3[i])
    fig.data[i].marker.color = tuple(colors)
# Change the bar mode
fig.update_layout(barmode='group',
                  width=400,
                  height=400,
                  xaxis=dict(showgrid=False, ticktext=["log<sub>10</sub> PLD",
                                                       "log<sub>10</sub> LCD",
                                                       "Density",
                                                       "VSA",
                                                       "VF",
                                                       "Qst CH<sub>4</sub>",
                                                       "Qst CO<sub>2</sub>",
                                                       "Qst H<sub>2</sub>S",
                                                       "Qst H<sub>2</sub>O"]),
                  yaxis=dict(showgrid=False),
                  yaxis_title="Importance",
                  plot_bgcolor='white',
                  margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(yanchor="top", y=0.995, xanchor="right", x=0.995))
fig.update_xaxes(
    tickangle=90,
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    title_standoff=5,
)
fig.update_yaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    title_standoff=3,
)
filename = "../Graphs/ML_graphs/Regression/RF_importance.png"
fig.write_image(filename, scale=2)
