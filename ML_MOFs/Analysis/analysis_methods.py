import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import math
import plotly.graph_objects as go


# calculate the min, max, mean, median, and std of data
def data_analysis_table(df):
    stats = []
    for desc in list(df.columns):
        temp = [desc, min(df[desc]), max(df[desc]), np.mean(df[desc].tolist()), np.median(df[desc].tolist()),
                np.std(df[desc].tolist())]
        stats.append(temp)
    stats = pd.DataFrame(data=stats, columns=["Descriptor", "Minimum", "Maximum", "Mean", "Median", "Std Dev"])
    stats = stats.round(3)
    return stats


def range_subplot(df, file_name):
    descriptor_names = list(df.columns)
    title_names = []
    for f in descriptor_names:
        title_names.append(f)
    for n in range(len(title_names)):
        if len(title_names[n]) > 25:
            title_names[n] = title_names[n][0:24]
    fig = make_subplots(rows=math.ceil(len(descriptor_names) / 2), cols=2, subplot_titles=title_names,
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for desc in range(len(descriptor_names)):
        if (desc + 1) % 2 == 1:
            fig.add_trace(go.Histogram(x=df[descriptor_names[desc]], marker=dict(color="#1f77b4")),
                          row=math.ceil((desc + 1) / 2), col=1)
        if (desc + 1) % 2 == 0:
            fig.add_trace(go.Histogram(x=df[descriptor_names[desc]], marker=dict(color="#1f77b4")),
                          row=math.ceil((desc + 1) / 2), col=2)
    fig.update_layout(width=600, height=600, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    fig.update_traces(hovertemplate="Range: %{x}<br>Frequency: %{y}<extra></extra>")
    fig.update_annotations(font_size=11)
    filename = "../Graphs/Analysis_graphs/" + file_name + ".png"
    fig.write_image(filename, scale=2)
