# run random forest models removing one descriptor in turn to assess descriptor importance
from ML_methods import regression
import pandas as pd
from ML_main import final_descriptors, regression_targets
import plotly.graph_objs as go
import plotly.express as px


data = pd.read_csv("..\\Data\\MOF_data.csv")

# regression
ML_metrics = []
for target in regression_targets:
    for desc in final_descriptors:
        descriptors = [n for n in final_descriptors if n != desc]
        print(target)
        print(descriptors)
        predictions, metrics, importance, _ = regression(data, descriptors, target, "RF")
        metrics.insert(2, desc)
        ML_metrics.append(metrics)
ML_metrics = pd.DataFrame(data=ML_metrics, columns=["Target", "Method", "Descriptor Removed", "Mean R2", "SD R2",
                                                    "Mean MAE", "SD MAE", "Target SD"])
ML_metrics.to_csv("..\\Results\\ML_results\\regression\\loo_RF_metrics.csv")

# plot results
axis_titles = {
    "CO2 loading (mol/kg)": "BM CO<sub>2</sub> Loading",
    "CH4 loading (mol/kg)": "BM CH<sub>4</sub> Loading",
    "SC CO2 loading (mol/kg)": "SC CO<sub>2</sub> Loading",
    "SC CH4 loading (mol/kg)": "SC CH<sub>4</sub> Loading",
    "TSN": "TSN",
    "LOG10 TSN": "log<sub>10</sub> TSN"
}

df = pd.read_csv("..\\Results\\ML_results\\regression\\loo_RF_metrics.csv")

metrics = ["MAE", "R2"]
for met in metrics:
    if met == "R2":
        yaxis_title = "Mean R<sup>2</sup>"
    else:
        yaxis_title = "Mean MAE"
    for tar in regression_targets:
        df_data = df[df["Target"] == tar]
        fig = px.bar(df_data, x="Descriptor Removed", y="Mean " + met, error_y="SD " + met)
        fig.update_layout(barmode='group',
                          width=400,
                          height=400,
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False),
                          yaxis_title=yaxis_title,
                          plot_bgcolor='white',
                          title=axis_titles[tar],
                          margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(yanchor="top", y=0.995, xanchor="right", x=0.995))
        fig.update_xaxes(
            tickangle=90,
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            title_standoff=5,
            ticktext=["log<sub>10</sub> PLD",
                      "log<sub>10</sub> LCD",
                      "Density",
                      "VSA",
                      "VF",
                      "Qst CH<sub>4</sub>",
                      "Qst CO<sub>2</sub>",
                      "Qst H<sub>2</sub>S",
                      "Qst H<sub>2</sub>O"],
            tickvals=df_data["Descriptor Removed"]
        )
        fig.update_yaxes(
            mirror=True,
            ticks='outside',
            showline=True,
            linecolor='black',
            title_standoff=3
        )
        tar = tar.replace("(", "_")
        tar = tar.replace(")", "_")
        tar = tar.replace("/", "_")
        tar = tar.replace(" ", "_")
        filename = "../Graphs/ML_graphs/LOO_RF_Regression/" + met + "_" + tar + ".png"
        fig.write_image(filename, scale=2)
