import plotly.graph_objects as go

# figure 2
fig = go.Figure(data=[go.Table(
    columnwidth=[300, 300, 300],
    header=dict(values=['', '<b>HIGH (SD)</b>', '<b>LOW (SD)</b>'],
                line_color='black',
                fill_color='white',
                align='left',
                height=40,
                font=dict(size=22),
                line=dict(width=3)
                ),
    cells=dict(values=[["Precision", "Recall", "F1 Score", "", "Accuracy", "Brier Score"],
                       ["0.866 (0.034)", "0.904 (0.037)", "0.884 (0.033)", "<b>Value (SD)</b>", "0.881 (0.027)", "0.083 (0.012)"],
                       ["0.898 (0.028)", "0.856 (0.026)", "0.876 (0.022)"]],
               line_color='black',
               fill_color='white',
               align='left',
               height=40,
               line=dict(width=3)))
])

fig.update_layout(width=800, height=400, margin=dict(l=20, r=20, t=20, b=55))
fig.update_traces(cells_font=dict(size=22))

fig.write_image("Graphs/Figures/Figure 2/table.png", scale=1)

fig = go.Figure(data=[go.Table(
    columnwidth=[300, 200, 200],
    header=dict(values=['', '<b>HIGH</b>', '<b>LOW</b>'],
                line_color='black',
                fill_color='white',
                align='left',
                height=40,
                font=dict(size=22),
                line=dict(width=3)
                ),
    cells=dict(values=[["Precision", "Recall", "F1 Score", "", "Accuracy", "Brier Score"],
                       ["0.774", "0.372", "0.503", "<b>Value</b>", "0.712", "0.168"],
                       ["0.698", "0.930", "0.797"]],
               line_color='black',
               fill_color='white',
               align='left',
               height=40,
               line=dict(width=3)))
])

fig.update_layout(width=400, height=400, margin=dict(l=20, r=20, t=20, b=55))
fig.update_traces(cells_font=dict(size=22))

fig.write_image("Graphs/Figures/Figure 3/table.png", scale=1)
