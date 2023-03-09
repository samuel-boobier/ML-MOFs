from ML_methods import get_predictions
from graphs import get_graph
import numpy as np

# try best combination from initial screen with the full dataset
# 12 features (Spearman's rank), quantile uniform scaling, GPR, 10-fold cross validation
# try with TSN
# predictions = get_predictions("TSN")
# get_graph(predictions, "TSN")
# try with LOG10 TSN
predictions = get_predictions("LOG10 TSN")
get_graph(predictions, "LOG10 TSN")
# un-log LOG10 TSN predictions
predictions["LOG10 TSN"] = np.power(10, predictions["LOG10 TSN"].values)
predictions["Prediction"] = np.power(10, predictions["Prediction"].values)
get_graph(predictions, "LOG10 TSN")
