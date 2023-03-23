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
