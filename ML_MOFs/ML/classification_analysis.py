import pandas as pd
import plotly.express as px


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

# need to average out rocs
# df_roc = pd.read_csv("..\\Results\\ML_results\\Classification\\RF_roc.csv")
# fig_thresh = px.line(
#                 df_roc, title='TPR and FPR at every threshold',
#                 width=700, height=500
#             )
# fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
# fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
# fig_thresh.show()


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
