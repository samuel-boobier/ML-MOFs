# run random forest models removing one descriptor in turn to assess descriptor importance
from ML_methods import regression
import pandas as pd
from ML_main import final_descriptors, regression_targets

data = pd.read_csv("..\\Data\\MOF_data.csv")

if __name__ == '__main__':
    # regression
    ML_metrics = []
    for target in regression_targets:
        for desc in final_descriptors:
            descriptors = [n for n in final_descriptors if n != desc]
            predictions, metrics, importance, _ = regression(data, descriptors, target, "RF")
            ML_metrics.insert(2, desc)
            ML_metrics.append(metrics)
    ML_metrics = pd.DataFrame(data=ML_metrics, columns=["Target", "Method", "Descriptor Removed", "Mean R2", "SD R2",
                                                        "Mean MAE", "SD MAE", "Target SD"])
    ML_metrics.to_csv("..\\Results\\ML_results\\regression\\loo_RF_metrics.csv")
