import pandas as pd

from ML_new.ML_methods import get_predictions, bin_data
from ML_new.graphs import get_graph, plot_distribution
import numpy as np

# bin_data("TSN")
# dealing with skewed target variable TSN
# Use RF and 12 features selected by initial screen with standardisation
# 1 - don't deal with skewed data
# predictions_1 = get_predictions("TSN")
# plot_distribution(predictions_1, "TSN")
# get_graph(predictions_1, "TSN")
# 2 - manually even out data by binning in 1 TSN intervals and taking equal portions of each
# 3 - SMOTER or SMOGN to automatically deal with skew automatically
# 4 - transform data with LOG10, cube root, etc.
# LOG10
# predictions_1 = get_predictions("LOG10 TSN")
# plot_distribution(predictions_1, "LOG10 TSN")
# get_graph(predictions_1, "TSN")
# ROOT 3
# predictions_1 = get_predictions("ROOT3 TSN")
# plot_distribution(predictions_1, "ROOT3 TSN")
# get_graph(predictions_1, "TSN")

# prediction CO2 and CH4 loading, then calculate TSN
# get CO2 predictions
# predictions_CO2 = get_predictions("CO2 loading (mol/kg)")
# # get CH4 predictions
# predictions_CH4 = get_predictions("CH4 loading (mol/kg)")
# # plot distribution of CO2 and CH4
# plot_distribution(predictions_CO2, "CO2 loading (mol/kg)")
# plot_distribution(predictions_CH4, "CH4 loading (mol/kg)")
# # plot predictions of CO2 and CH4
# get_graph(predictions_CO2, "CO2 loading (mol/kg)")
# get_graph(predictions_CH4, "CH4 loading (mol/kg)")
# # calculate selectivity prediction
# selectivity_prediction = np.array(predictions_CO2["Prediction"].tolist())/np.array(predictions_CH4["Prediction"].tolist())
# TSN_preds = np.array(predictions_CO2["Prediction"].tolist()) * np.log10(selectivity_prediction)
# # re-form TSN
# selectivity = np.array(predictions_CO2["CO2 loading (mol/kg)"].tolist())/np.array(predictions_CH4["CH4 loading (mol/kg)"].tolist())
# TSN = np.array(predictions_CO2["CO2 loading (mol/kg)"].tolist()) * np.log10(selectivity)
# # TSN predictions
# predictions_TSN = pd.DataFrame(data=np.array([predictions_CO2["MOF"].tolist(), TSN, TSN_preds]).T,
#                                columns=["MOF", "TSN", "Prediction"])
# predictions_TSN = predictions_TSN.astype({"TSN": 'float', 'Prediction': 'float'})
# # plot TSN predictions
# get_graph(predictions_TSN, "TSN")
# classification model
preds = get_predictions("TSN Class", True)
# get_graph(preds, "CO2 loading (mol/kg)")

