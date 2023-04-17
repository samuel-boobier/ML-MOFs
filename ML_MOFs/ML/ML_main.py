from ML_methods import regression, classification
import pandas as pd
from ML_graphs import get_graph

data = pd.read_csv("..\\Data\\MOF_data.csv")
final_descriptors = ["PLD log10", "LCD log10", "Density (g/cc)", "VSA (m2/cc)", "VF", "Qst_CH4", "Qst_CO2", "Qst_H2S",
                     "Qst_H2O"]
regression_targets = ["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "SC CO2 loading (mol/kg)",
                      "SC CH4 loading (mol/kg)", "TSN", "LOG10 TSN"]
intervals = {
    "CO2 loading (mol/kg)": 5,
    "CH4 loading (mol/kg)": 0.5,
    "SC CO2 loading (mol/kg)": 5,
    "SC CH4 loading (mol/kg)": 2,
    "TSN": 5,
    "LOG10 TSN": 0.5
}
SVM_parameters = {
    "CO2 loading (mol/kg)": {"C": 10, "epsilon": 0.1, "gamma": 0.1},
    "CH4 loading (mol/kg)": {"C": 10, "epsilon": 0.1, "gamma": 0.1},
    "SC CO2 loading (mol/kg)": {"C": 1000, "epsilon": 0.1, "gamma": 0.01},
    "SC CH4 loading (mol/kg)": {"C": 10, "epsilon": 0.1, "gamma": 0.1},
    "TSN": {"C": 10, "epsilon": 0.1, "gamma": "auto"},
    "LOG10 TSN": {"C": 100, "epsilon": 0.1, "gamma": 0.01}
}

# # regression
# # ML_methods = ["RF", "MLR", "SVM"]
# ML_methods = ["RF", "MLR", "SVM"]
# for method in ML_methods:
#     ML_metrics = []
#     ML_preds = data[["MOF"]]
#     importance_ls = pd.DataFrame(final_descriptors, columns=["Descriptor"])
#     for target in regression_targets:
#         if method == "SVM":
#             predictions, metrics, _ = regression(data, final_descriptors, target, method, SVM_parameters[target]["C"],
#                                                  SVM_parameters[target]["epsilon"], SVM_parameters[target]["gamma"])
#         elif method == "RF":
#             predictions, metrics, importance = regression(data, final_descriptors, target, method)
#             importance_ls = pd.merge(importance_ls, importance, on=["Descriptor", "Descriptor"])
#         else:
#             predictions, metrics, _ = regression(data, final_descriptors, target, method)
#         get_graph(predictions, target, method, intervals[target])
#         ML_metrics.append(metrics)
#         ML_preds = pd.merge(ML_preds, predictions, on=["MOF", "MOF"])
#     ML_metrics = pd.DataFrame(data=ML_metrics, columns=["Target", "Method", "Mean R2", "SD R2", "Mean MAE", "SD MAE",
#                                                         "Target SD"])
#     ML_metrics.to_csv("..\\Results\\ML_results\\regression\\" + method + "_metrics.csv")
#     ML_preds.to_csv("..\\Results\\ML_results\\regression\\" + method + "_predictions.csv")
#     if method == "RF":
#         importance_ls.to_csv("..\\Results\\ML_results\\regression\\" + method + "_importance.csv")
#
# # classification
# # ML_methods = ["RF", "SVM", "KNN"]
# ML_methods = ["RF", "SVM", "KNN"]
# target = "TSN Class"
# for method in ML_methods:
#     classification(data, final_descriptors, target, method)
