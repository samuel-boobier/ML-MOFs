from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# defining parameter range
# regression SVM
param_grid_svm = {
    'svr__C': [0.001, 0.1, 1, 10, 100, 1000],
    'svr__gamma': [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
    'svr__kernel': ['rbf'],
    'svr__epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
}
param_grid_knn = {'kneighborsclassifier__n_neighbors': np.arange(1, 100)}
param_grid_svm_classification = {
    'svc__C': [0.001, 0.1, 1, 10, 100, 1000],
    'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
    'svc__kernel': ['rbf'],
}


def get_parameters(X, y, model, method, ML_type):
    if ML_type == "regression":
        param_grid = param_grid_svm
    elif method == "SVM":
        param_grid = param_grid_svm_classification
    else:
        param_grid = param_grid_knn
    pipe = make_pipeline(StandardScaler(), model)
    grid = GridSearchCV(pipe, param_grid, refit=True, verbose=0)
    grid.fit(X, y)
    return grid.best_params_


if __name__ == '__main__':

    data = pd.read_csv("..\\Data\\MOF_data.csv")

    final_descriptors = ["PLD log10", "LCD log10", "Density (g/cc)", "VSA (m2/cc)", "VF", "Qst_CH4", "Qst_CO2", "Qst_H2S",
                         "Qst_H2O"]
    regression_targets = ["CO2 loading (mol/kg)", "CH4 loading (mol/kg)", "SC CO2 loading (mol/kg)",
                          "SC CH4 loading (mol/kg)", "TSN", "LOG10 TSN"]

    for target in regression_targets:
        X = data[final_descriptors]
        y = data[target].tolist()
        print(get_parameters(X, y, SVR(), "SVM", "regression"))

    classification_target = "TSN Class"
    X = data[final_descriptors]
    y = data[classification_target].tolist()
    print(get_parameters(X, y, KNeighborsClassifier(), "KNN", "classification"))
    print(get_parameters(X, y, SVC(), "SVM", "classification"))
