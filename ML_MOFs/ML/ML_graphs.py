import plotly.express as px
import numpy as np


def get_graph(data, target, method, interval):
    fig = px.scatter(data, x=target, y=target + " Prediction", hover_name="MOF", title=target + " - " + method)
    fig.update_layout(width=600, height=600)
    fig.update_annotations(font_size=8)
    max_number = max(data[target].tolist() + data[target + " Prediction"].tolist())
    max_number += interval*0.2
    min_number = min(data[target].tolist() + data[target + " Prediction"].tolist())
    min_number -= interval*0.2
    fig.update_yaxes(range=[min_number, max_number])
    fig.update_xaxes(range=[min_number, max_number])
    fig.update_layout(
        xaxis=dict(
            dtick=interval
        ),
        yaxis=dict(
            dtick=interval
        )
    )
    target = target.replace("(", "_")
    target = target.replace(")", "_")
    target = target.replace("/", "_")
    filename = "..\\Graphs\\ML_graphs\\Regression\\" + method + "_" + target + ".png"
    fig.write_image(filename, scale=2)


def get_graph_test(data, target, method):
    fig = px.scatter(data, x="Real", y=method, hover_name="MOF", title=target + " - " + method)
    fig.update_layout(width=600, height=600)
    fig.update_annotations(font_size=8)
    max_number = max(data["Real"].tolist() + data[method].tolist())
    max_number += np.absolute(max_number*0.2)
    min_number = min(data["Real"].tolist() + data[method].tolist())
    min_number -= np.absolute(min_number*0.2)
    fig.update_yaxes(range=[min_number, max_number])
    fig.update_xaxes(range=[min_number, max_number])
    fig.update_layout(
        yaxis_title="Prediction",
        xaxis_title=target,
    )
    target = target.replace("(", "_")
    target = target.replace(")", "_")
    target = target.replace("/", "_")
    filename = "..\\Results\\ML_results\\Test_set\\" + method + "_" + target + ".png"
    fig.write_image(filename, scale=2)
