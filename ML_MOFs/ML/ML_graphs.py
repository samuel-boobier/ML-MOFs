import plotly.express as px


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
    filename = "../Graphs/ML_graphs/Regression/" + method + "_" + target + ".png"
    fig.write_image(filename, scale=2)
