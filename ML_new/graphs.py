import plotly.express as px


def get_graph(data, TSN):
    fig = px.scatter(data, x=TSN, y="Prediction", hover_name="MOF", title="MOF TSN Prediction")
    fig.show()


def plot_distribution(data, TSN):
    fig = px.histogram(data, x=TSN, marginal="box")
    fig.show()
