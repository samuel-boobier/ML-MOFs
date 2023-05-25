import pandas as pd
import plotly.express as px
from sklearn.metrics import auc
import numpy as np


# plot histogram of HIGH and LOW predictions compared to TSN
def get_classification_histogram(method):
    # change here for training results
    prediction_data = pd.read_csv("../Results/ML_results/Test_set/" + method + "_classification_predictions.csv")
    prediction_data = prediction_data.sort_values(by="MOF")
    data = pd.read_csv("..\\Data\\MOF_data_test.csv")
    data = data.sort_values(by="MOF")
    prediction_data["TSN"] = data["TSN"].tolist()
    fig = px.histogram(prediction_data, x="TSN", color="TSN Class Prediction", barmode="overlay", width=600, height=200)
    fig.add_vline(x=5, line_color="black", line_width=1, line_dash="dash")
    fig.update_layout(yaxis=dict(showgrid=False),
                      yaxis_title="Frequency",
                      plot_bgcolor='white',
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(yanchor="top", y=0.995, xanchor="right", x=0.995))
    fig.update_xaxes(
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
        title_standoff=3
    )
    # change here for training results
    filename = "../Results/ML_results/Test_set/" + method + "_histogram.png"
    fig.write_image(filename, scale=2)


get_classification_histogram("RF")
get_classification_histogram("SVM")
get_classification_histogram("KNN")


# roc plots
def roc_plot(method):
    # change here for training results
    df_roc = pd.read_csv("../Results/ML_results/Test_set/" + method + "_roc.csv")

    fpr = df_roc["False Positive Rate"].tolist()
    tpr = df_roc["True Positive Rate"].tolist()

    fig = px.area(
        x=fpr, y=tpr,
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=200, height=200, color_discrete_sequence=["black"]
    )
    fig.add_shape(
        type='line', line=dict(dash='dash', color="black", width=1),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig.update_yaxes(mirror=True,
                     ticks='outside',
                     showline=True,
                     linecolor='black',
                     title_standoff=5,
                     range=[0, 1]
                     )
    fig.update_xaxes(mirror=True,
                     ticks='outside',
                     showline=True,
                     linecolor='black',
                     title_standoff=5,
                     range=[0, 1]
                     )
    fig.update_traces(line=dict(width=1))
    filename = "../Results/ML_results/Test_set/" + method + "_roc_curve.png"
    print(auc(fpr, tpr))
    fig.write_image(filename, scale=2)


roc_plot("RF")
roc_plot("SVM")
roc_plot("KNN")
#
# probability plots
target = "TSN Class"


def probability_plots(method):
    # change here for training results
    df = pd.read_csv("../Results/ML_results/Test_set/" + method + "_classification_predictions.csv")
    # The histogram of scores compared to true labels
    fig = px.histogram(df, x="HIGH Probability", color="TSN Class", nbins=50, width=600, height=200,
                       labels=dict(color='True Labels', x='Score'), barmode="overlay")
    fig.update_layout(yaxis=dict(showgrid=False),
                      yaxis_title="Frequency",
                      plot_bgcolor='white',
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(yanchor="top", y=0.995, xanchor="right", x=0.995))
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        title_standoff=5,
        range=[0, 1],
        constrain='domain'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        title_standoff=3,
    )
    # change here for training results
    filename = "../Results/ML_results/Test_set/" + method + "_probs_histogram.png"
    fig.write_image(filename, scale=2)


probability_plots("RF")
probability_plots("SVM")
probability_plots("KNN")

# importance plot
descs = {
    "PLD log10": "log<sub>10</sub> PLD",
    "LCD log10": "log<sub>10</sub> LCD",
    "Density (g/cc)": "Density",
    "VSA (m2/cc)": "VSA",
    "VF": "VF",
    "Qst_CH4": "Qst CH<sub>4</sub>",
    "Qst_CO2": "Qst CO<sub>2</sub>",
    "Qst_H2S": "Qst H<sub>2</sub>S",
    "Qst_H2O": "Qst H<sub>2</sub>O"
}


# change here for training results
data = pd.read_csv("..\\Results\\ML_results\\Classification\\RF_importance.csv")
data = data.sort_values(by="TSN Class Mean", ascending=False)
fig = px.bar(data, x='Descriptor', y='TSN Class Mean', error_y='TSN Class SD', color_discrete_sequence=[px.colors.
             qualitative.D3[0]])
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
fig.update_layout(barmode='group',
                  xaxis_title=None,
                  width=200,
                  height=200,
                  xaxis=dict(showgrid=False, tickvals=np.arange(0, len(descs)), ticktext=[descs[x] for x in
                                                                                          data["Descriptor"]]),
                  yaxis=dict(showgrid=False),
                  yaxis_title="Importance",
                  plot_bgcolor='white',
                  margin=dict(l=10, r=10, t=10, b=0),
                  )
fig.data[0].error_y.thickness = 1
fig.data[0].error_y.width = 2
filename = "../Graphs/ML_graphs/Classification/RF_importance.png"
fig.write_image(filename, scale=2)
