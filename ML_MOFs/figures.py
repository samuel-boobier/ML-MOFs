import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

# Figures for the publication
from sklearn.metrics import r2_score

df_data = pd.read_csv("Data/MOF_data.csv")

# Figure 1b
# Ranges of target values as %s
# targets = ["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "SC CO2 loading (mol/kg)", "SC CH4 loading (mol/kg)", "TSN",
#            "LOG10 TSN"]
# target_ranges = {
#     "CO2 loading (mol/kg)": np.arange(0, 20, 5),
#     "CH4 loading (mol/kg)": np.arange(0, 4, 1),
#     "SC CO2 loading (mol/kg)": np.arange(0, 25, 5),
#     "SC CH4 loading (mol/kg)": np.arange(0, 12, 4),
#     "TSN": np.arange(0, 40, 10),
#     "LOG10 TSN": np.arange(-1, 2, 1)
# }
#
# target_max = {
#     "CO2 loading (mol/kg)": [0, 17.5],
#     "CH4 loading (mol/kg)": [0, 3.65],
#     "SC CO2 loading (mol/kg)": [0, 24.5],
#     "SC CH4 loading (mol/kg)": [0, 10.6],
#     "TSN": [0, 33],
#     "LOG10 TSN": [-1.3, 1.55]
# }
#
# axis_titles = {
#     "CO2 loading (mol/kg)": "BM CO<sub>2</sub> Loading",
#     "CH4 loading (mol/kg)": "BM CH<sub>4</sub> Loading",
#     "SC CO2 loading (mol/kg)": "SC CO<sub>2</sub> Loading",
#     "SC CH4 loading (mol/kg)": "SC CH<sub>4</sub> Loading",
#     "TSN": "TSN",
#     "LOG10 TSN": "log<sub>10</sub> TSN"
# }
#
#
# def target_histograms(data, desc):
#     fig = px.histogram(data, x=desc, histnorm='percent', width=200, height=200, color_discrete_sequence=["black"])
#     fig.update_layout(
#         xaxis=dict(showgrid=False, tickvals=target_ranges[desc]),
#         yaxis=dict(showgrid=False),
#         yaxis_title="Frequency / %",
#         xaxis_title=axis_titles[desc],
#         plot_bgcolor='white',
#         margin=dict(l=20, r=20, t=20, b=20)
#     )
#     fig.update_xaxes(
#         mirror=True,
#         ticks='outside',
#         showline=True,
#         linecolor='black',
#         range=target_max[desc],
#         title_standoff=5
#     )
#     fig.update_yaxes(
#         mirror=True,
#         ticks='outside',
#         showline=True,
#         linecolor='black',
#         title_standoff=5
#     )
#     if desc == "TSN":
#         fig.add_vline(x=5, line_color="red", line_width=1.5)
#     desc = desc.replace("(", "_")
#     desc = desc.replace(")", "_")
#     desc = desc.replace("/", "_")
#     filename = "Graphs/Figures/Figure 1/" + desc + ".png"
#     fig.write_image(filename, scale=2)
#
#
# for t in targets:
#     target_histograms(df_data, t)

# uptake vs selectivity graphs
# df_data["Selectivity (CO2)"] = df_data["Selectivity (CO2)"].div(1000)
# df_data["log10 Selectivity (CO2)"] = np.log10(np.array(df_data["Selectivity (CO2)"].tolist()))
# fig = px.scatter(df_data, x="CO2 loading (mol/kg)", y="Selectivity (CO2)", width=200, height=200,
#                  color_discrete_sequence=["black"])
# fig.update_layout(
#     xaxis=dict(showgrid=False),
#     yaxis=dict(showgrid=False),
#     yaxis_title='CO<sub>2</sub> Selectivity <span>&#215;</span> 10<sup>-3</sup>',
#     xaxis_title="BM CO<sub>2</sub> Loading",
#     plot_bgcolor='white',
#     margin=dict(l=20, r=20, t=20, b=20)
# )
# fig.update_xaxes(
#     mirror=True,
#     ticks='outside',
#     showline=True,
#     linecolor='black',
#     title_standoff=5,
#     range=[0, 19]
# )
# fig.update_yaxes(
#     mirror=True,
#     ticks='outside',
#     showline=True,
#     linecolor='black',
#     title_standoff=3,
#     range=[0, 14]
# )
# fig.update_traces(marker=dict(size=4))
# filename = "Graphs/Figures/Figure 1/BM_CO2_Selectivity.png"
# fig.write_image(filename, scale=2)
#
# fig = px.scatter(df_data, x="CO2 loading (mol/kg)", y="log10 Selectivity (CO2)", width=200, height=200,
#                  color_discrete_sequence=["black"])
# fig.update_layout(
#     xaxis=dict(showgrid=False),
#     yaxis=dict(showgrid=False),
#     yaxis_title="log<sub>10</sub> CO<sub>2</sub> Selectivity",
#     xaxis_title="BM CO<sub>2</sub> Loading",
#     plot_bgcolor='white',
#     margin=dict(l=20, r=20, t=20, b=20)
# )
# fig.update_xaxes(
#     mirror=True,
#     ticks='outside',
#     showline=True,
#     linecolor='black',
#     title_standoff=5,
# )
# fig.update_yaxes(
#     mirror=True,
#     ticks='outside',
#     showline=True,
#     linecolor='black',
#     title_standoff=3,
# )
# fig.update_traces(marker=dict(size=4))
# filename = "Graphs/Figures/Figure 1/log10_BM_CO2_Selectivity.png"
# fig.write_image(filename, scale=2)

# Figure 2
# a) correlation analysis
# descriptors = ["TSN", "LOG10 TSN", "SC CO2 loading (mol/kg)", "CO2 loading (mol/kg)", "SC CH4 loading (mol/kg)",
#                "CH4 loading (mol/kg)", "PLD log10", "LCD log10", "Density (g/cc)", "VSA (m2/cc)", "GSA (m2/g)",
#                "VF", "PV (cc/g) log10", "K0_CH4 log10", "K0_CO2 log10", "K0_H2S log10", "K0_H2O log10", "Qst_CO2",
#                "Qst_H2S", "Qst_H2O"]
# Data = df_data[descriptors]
# # get correlation table
# corr = Data.corr()
# # convert to R^2
# corr = corr.pow(2)
# col_names = Data.dtypes.index
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr, cmap='Reds', vmin=0, vmax=1)
# cb = fig.colorbar(cax, shrink=0.75)
# cb.ax.tick_params(labelsize=12)
# ticks = np.arange(0, len(Data.columns), 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(np.arange(1, len(col_names) + 1), fontsize=12)
# ax.set_yticklabels(np.arange(1, len(col_names) + 1), fontsize=12)
# plt.ylim(len(Data.columns)-0.5, -0.5)
# plt.tight_layout()
# filename = "Graphs/Figures/Figure 2/correlation.png"
# plt.savefig(filename, dpi=600)
# b) SC CO2 loading vs VF
fig = px.scatter(df_data, x="SC CO2 loading (mol/kg)", y="VF", width=200, height=200,
                 color_discrete_sequence=["black"])
fig.update_layout(
    xaxis=dict(showgrid=False, tickvals=np.arange(0, 30, 5)),
    yaxis=dict(showgrid=False),
    yaxis_title='VF',
    xaxis_title="SC CO<sub>2</sub> Loading",
    plot_bgcolor='white',
    margin=dict(l=20, r=20, t=20, b=20)
)
fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    title_standoff=5,
    range=[0, 25]
)
fig.update_yaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    title_standoff=3,
    range=[0.2, 1]
)
fig.update_traces(marker=dict(size=4))
filename = "Graphs/Figures/Figure 2/SC_CO2_VF.png"
fig.write_image(filename, scale=2)
# c) - h) RF model 10-CV for regression targets
# i) error vs prediction TSN
# j) classification ROC
# k) classification TSN vs prediction
# l) classification certainty vs class
# m) feature importance plots

# Table 2 - Metrics all models

# Figure 3 - External test set