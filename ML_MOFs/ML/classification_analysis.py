import pandas as pd
import plotly.express as px
from sklearn.metrics import auc


# plot histogram of HIGH and LOW predictions compared to TSN
def get_classification_histogram(method):
    prediction_data = pd.read_csv("..\\Results\\ML_results\\Classification\\" + method + "_predictions.csv")
    prediction_data = prediction_data.sort_values(by="MOF")
    data = pd.read_csv("..\\Data\\MOF_data.csv")
    data = data.sort_values(by="MOF")
    prediction_data["TSN"] = data["TSN"].tolist()
    fig = px.histogram(prediction_data, x="TSN", color="TSN Class Prediction", barmode="overlay", width=600, height=400,
                       title="TSN Class Predictions vs TSN - " + method)
    fig.add_vline(x=5, line_color="black", line_width=1, line_dash="dash")
    fig.update_yaxes(range=[0, 150])
    filename = "../Graphs/ML_graphs/Classification/" + method + "_histogram.png"
    fig.write_image(filename, scale=2)


get_classification_histogram("RF")
get_classification_histogram("SVM")
get_classification_histogram("KNN")


# roc plots
def roc_plot(method):
    df_roc = pd.read_csv("..\\Results\\ML_results\\Classification\\" + method + "_roc.csv")

    fpr = df_roc["False Positive Rate"].tolist()
    tpr = df_roc["True Positive Rate"].tolist()

    fig = px.area(
        x=fpr, y=tpr,
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=400, height=400
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(title_text='ROC Curve - ' + method + f' (AUC={auc(fpr, tpr):.4f})', title_x=0.5)
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    filename = "../Graphs/ML_graphs/Classification/" + method + "_roc_curve.png"
    fig.write_image(filename, scale=2)


roc_plot("RF")
roc_plot("SVM")
roc_plot("KNN")

# probability plots
target = "TSN Class"


def probability_plots(method):
    df = pd.read_csv("..\\Results\\ML_results\\Classification\\" + method + "_predictions.csv")
    # The histogram of scores compared to true labels
    fig_hist = px.histogram(df, x="HIGH Probability", color="TSN Class", nbins=50, width=600, height=400,
                            labels=dict(color='True Labels', x='Score'), barmode="overlay",
                            title="TSN Class Probability vs TSN Class - " + method)
    fig_hist.update_xaxes(range=[0, 1], constrain='domain')
    filename = "../Graphs/ML_graphs/Classification/" + method + "_probs_histogram.png"
    fig_hist.write_image(filename, scale=2)


probability_plots("RF")
probability_plots("SVM")
probability_plots("KNN")

# importance plot

data = pd.read_csv("..\\Results\\ML_results\\Classification\\RF_importance.csv")
data = data.sort_values(by="TSN Class Mean", ascending=False)
fig = px.bar(data, x='Descriptor', y='TSN Class Mean', error_y='TSN Class SD', width=600, height=500,
             title="Feature Importance - RF")
fig.update_xaxes(tickangle=90)
filename = "../Graphs/ML_graphs/Classification/RF_importance.png"
fig.write_image(filename, scale=2)
