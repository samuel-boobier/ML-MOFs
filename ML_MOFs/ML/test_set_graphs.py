from ML_graphs import get_graph_test
import pandas as pd
from ML_main import regression_targets

methods = ["MLR", "SVM", "RF"]
data = pd.read_csv("../Results/ML_results/Test_set/regression_predictions.csv")
for method in methods:
    for target in regression_targets:
        graph_data = data[data["Target"] == target]
        get_graph_test(graph_data, target, method)

