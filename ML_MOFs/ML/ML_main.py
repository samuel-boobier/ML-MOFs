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


if __name__ == '__main__':
    # regression
    # ML_methods = ["RF", "MLR", "SVM"]
    ML_methods = ["SVM"]
    for method in ML_methods:
        if method == "SVM":
            SVM_params = []
        ML_metrics = []
        ML_preds = data[["MOF"]]
        importance_ls = pd.DataFrame(final_descriptors, columns=["Descriptor"])
        for target in regression_targets:
            if method == "SVM":
                predictions, metrics, _, params_df = regression(data, final_descriptors, target, method)
                params_df["Target"] = [target for _ in range(params_df.shape[0])]
                SVM_params.append(params_df)
            elif method == "RF":
                predictions, metrics, importance, _ = regression(data, final_descriptors, target, method)
                importance_ls = pd.merge(importance_ls, importance, on=["Descriptor", "Descriptor"])
            else:
                predictions, metrics, _, _ = regression(data, final_descriptors, target, method)
            get_graph(predictions, target, method, intervals[target])
            ML_metrics.append(metrics)
            ML_preds = pd.merge(ML_preds, predictions, on=["MOF", "MOF"])
        ML_metrics = pd.DataFrame(data=ML_metrics, columns=["Target", "Method", "Mean R2", "SD R2", "Mean MAE", "SD MAE",
                                                            "Target SD"])
        ML_metrics.to_csv("..\\Results\\ML_results\\regression\\" + method + "_metrics.csv")
        ML_preds.to_csv("..\\Results\\ML_results\\regression\\" + method + "_predictions.csv")
        if method == "RF":
            importance_ls.to_csv("..\\Results\\ML_results\\regression\\" + method + "_importance.csv")
        if method == "SVM":
            SVM_params = pd.concat(SVM_params)
            SVM_params.to_csv("..\\Results\\Hyperparameters\\Regression\\" + method + "_hyperparameters.csv")


    # classification
    # ML_methods = ["RF", "SVM", "KNN"]
    ML_methods = ["KNN"]
    target = "TSN Class"
    for method in ML_methods:
        classification(data, final_descriptors, target, method)
