from ML_methods import regression
import pandas as pd
from ML_graphs import get_graph

data = pd.read_csv("..\\Data\\MOF_data.csv")
final_descriptors = ["PLD log10", "LCD log10", "Density (g/cc)", "VSA (m2/cc)", "VF", "DC_CH4 log10",
                     "DC_CO2 log10", "DC_H2S log10", "Qst_CH4", "Qst_CO2", "Qst_H2S", "Qst_H2O"]
regression_targets = ["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "SC CO2 loading (mol/kg)",
                      "SC CH4 loading (mol/kg)", "TSN", "LOG10 TSN"]
intervals = {"CO2 loading (mol/kg)": 5,
             "CH4 loading (mol/kg)": 0.5,
             "SC CO2 loading (mol/kg)": 5,
             "SC CH4 loading (mol/kg)": 2,
             "TSN": 5,
             "LOG10 TSN": 0.5}
# ML_methods = ["RF", "MLR", "SVM"]
ML_methods = ["SVM"]
for method in ML_methods:
    ML_metrics = []
    ML_preds = data[["MOF"]]
    for target in regression_targets:
        predictions, metrics = regression(data, final_descriptors, target, method)
        get_graph(predictions, target, method, intervals[target])
        ML_metrics.append(metrics)
        ML_preds = pd.merge(ML_preds, predictions, on=["MOF", "MOF"])
    ML_metrics = pd.DataFrame(data=ML_metrics, columns=["Target", "Method", "Mean R2", "SD R2", "Mean MAE", "SD MAE",
                                                        "Target SD"])
    ML_metrics.to_csv("..\\Results\\ML_results\\regression\\" + method + "_metrics.csv")
    ML_preds.to_csv("..\\Results\\ML_results\\regression\\" + method + "_predictions.csv")
