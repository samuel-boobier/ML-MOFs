import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# calculate the min, max, mean, median, and std of data
def data_analysis_table(df):
    stats = []
    for desc in list(df.columns):
        temp = [desc, min(df[desc]), max(df[desc]), np.mean(df[desc].tolist()), np.median(df[desc].tolist()),
                np.std(df[desc].tolist())]
        stats.append(temp)
    stats = pd.DataFrame(data=stats, columns=["Descriptor", "Minimum", "Maximum", "Mean", "Median", "Std Dev"])
    return stats


def range_subplot(df, file_name):
    descriptor_names = list(df.columns)
    title_names = []
    for f in descriptor_names:
        title_names.append(f)
    for n in range(len(title_names)):
        if len(title_names[n]) > 25:
            title_names[n] = title_names[n][0:24]

    fig = make_subplots(rows=math.ceil(len(descriptor_names) / 2), cols=math.ceil(len(descriptor_names) / 2),
                        subplot_titles=title_names, horizontal_spacing=0.06, vertical_spacing=0.12)
    for desc in range(len(descriptor_names)):
        if (desc + 1) % 2 == 1:
            fig.add_trace(go.Histogram(x=df[descriptor_names[desc]], marker=dict(color="#1f77b4")),
                          row=math.ceil((desc + 1) / 2), col=1)
        if (desc + 1) % 2 == 0:
            fig.add_trace(go.Histogram(x=df[descriptor_names[desc]], marker=dict(color="#1f77b4")),
                          row=math.ceil((desc + 1) / 2), col=2)
    if len(descriptor_names) > 2:
        w = 600
        h = 600
    elif len(descriptor_names) == 2:
        w = 600
        h = 300
    else:
        w = 300
        h = 300
    fig.update_layout(width=w, height=h, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    fig.update_traces(hovertemplate="Range: %{x}<br>Frequency: %{y}<extra></extra>")
    fig.update_annotations(font_size=11)
    filename = "../Graphs/Analysis_graphs/" + file_name + ".png"
    fig.write_image(filename, scale=2)


def corr_graph(Data, title, file_name):
    # get correlation table
    corr = Data.corr()
    # convert to R^2
    corr = corr.pow(2)
    col_names = Data.dtypes.index
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='Reds', vmin=0, vmax=1)
    cb = fig.colorbar(cax, shrink=0.825)
    cb.ax.tick_params(labelsize=9)
    ticks = np.arange(0, len(Data.columns), 1)
    ax.set_xticks(ticks)
    # plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(col_names, fontsize=8)
    ax.set_yticklabels(col_names, fontsize=8)
    ax.xaxis.set_tick_params(rotation=90)
    plt.ylim(len(Data.columns)-0.5, -0.5)
    # y parameter move title up to be legible
    ax.set_title(title, y=1.15, fontsize=12)
    # Find descriptors with R^2
    corr = Data.corr()
    corr = corr.pow(2)
    # using numpy, get values
    arr = corr.values
    index_names = corr.index
    col_names = corr.columns

    #  Get indices where such threshold is crossed; avoid diagonal elems
    R, C = np.where(np.triu(arr, 1) > 0)
    # Arrange those in columns and put out as a dataframe
    out_arr = np.column_stack((index_names[R], col_names[C], arr[R, C]))
    df_corr = pd.DataFrame(out_arr, columns=['row_name', 'col_name', 'R2'])
    df_corr = df_corr.sort_values(by='R2', ascending=False)
    plt.tight_layout()
    filename = "../Graphs/Analysis_graphs/" + file_name + ".png"
    plt.savefig(filename, dpi=600)
    return df_corr


def plot(n1, n2, data, name):
    plt.figure(figsize=(4.5, 4.5))
    x = list(data[n1])
    x = [float(i) for i in x]
    y = list(data[n2])
    y = [float(i) for i in y]
    plt.scatter(x, y, color="black", s=5)
    plt.title(n1 + " vs " + n2)
    plt.xlabel(n1)
    plt.ylabel(n2)
    plt.tight_layout()
    plt.savefig("../Graphs/Analysis_graphs/" + name + ".png", dpi=600)
