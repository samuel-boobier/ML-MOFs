from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# defining parameter range
# regression SVM
param_grid_svm = {
    'C': [0.001, 0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
    'kernel': ['rbf'],
    'epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
}
param_grid_knn = {'n_neighbors': np.arange(1, 100)}
param_grid_svm_classification = {
    'C': [0.001, 0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
    'kernel': ['rbf'],
}


def get_parameters(data, descriptors, target, model, param_grid):
    X = data[descriptors]
    y = data[target].tolist()
    # scale descriptors
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(data=X, columns=descriptors)
    grid = GridSearchCV(model, param_grid, refit=True, verbose=3)
    # fitting the model for grid search
    grid.fit(X, y)
    print(grid.best_params_)


data = pd.read_csv("..\\Data\\MOF_data.csv")

final_descriptors = ["PLD log10", "LCD log10", "Density (g/cc)", "VSA (m2/cc)", "VF", "DC_CH4 log10",
                     "DC_CO2 log10", "DC_H2S log10", "Qst_CH4", "Qst_CO2", "Qst_H2S", "Qst_H2O"]
# regression_targets = ["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "SC CO2 loading (mol/kg)",
#                       "SC CH4 loading (mol/kg)", "TSN", "LOG10 TSN"]
#
# for target in regression_targets:
#     get_parameters(data, final_descriptors, target, SVR(), param_grid_svm)

classification_target = "TSN Class"
# get_parameters(data, final_descriptors, classification_target, KNeighborsClassifier(), param_grid_knn)
get_parameters(data, final_descriptors, classification_target, SVC(), param_grid_svm_classification)
