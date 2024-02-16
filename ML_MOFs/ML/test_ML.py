import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, brier_score_loss, roc_curve
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from ML_main import final_descriptors, regression_targets

SVM_parameters = {
    "CO2 loading (mol/kg)": {'C': 10, 'epsilon': 0.1, 'gamma': 'scale'},
    "CH4 loading (mol/kg)": {'C': 10, 'epsilon': 0.1, 'gamma': 0.1},
    "SC CO2 loading (mol/kg)": {'C': 1000, 'epsilon': 1, 'gamma': 0.01, 'kernel': 'rbf'},
    "SC CH4 loading (mol/kg)": {'C': 10, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'},
    "TSN": {'C': 10, 'epsilon': 0.1, 'gamma': 'scale', 'kernel': 'rbf'},
    "LOG10 TSN": {'C': 1000, 'epsilon': 0.1, 'gamma': 0.01}
}


def test_model_regression(train, test, descriptors, target, C=None, epsilon=None, gamma=None):
    X_train = train[descriptors]
    y_train = train[target]
    X_test = test[descriptors]
    y_test = test[target]
    # scale based on training only
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # set up lists
    metrics = []
    predictions = [test["MOF"], [target for _ in range(len(y_test))], y_test]
    # MLR
    # set up and train model
    model = LinearRegression()
    model.fit(X_train, y_train.values.ravel())
    # get predictions
    preds = model.predict(X_test)
    # compute metrics
    metrics.append([target, "MLR", mean_absolute_error(y_test, preds), r2_score(y_test, preds)])
    predictions.append(preds)
    # SVM
    # set up and train model
    model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train, y_train.values.ravel())
    # get predictions
    preds = model.predict(X_test)
    # compute metrics
    metrics.append([target, "SVM", mean_absolute_error(y_test, preds), r2_score(y_test, preds)])
    predictions.append(preds)
    # RF
    # set up and train model
    model = RandomForestRegressor(n_estimators=500)
    model.fit(X_train, y_train.values.ravel())
    # get predictions
    preds = model.predict(X_test)
    # compute metrics
    metrics.append([target, "RF", mean_absolute_error(y_test, preds), r2_score(y_test, preds)])
    predictions.append(preds)
    return metrics, predictions


def test_model_classification(train, test, descriptors, target, method):
    X_train = train[descriptors]
    y_train = train[target]
    X_test = test[descriptors]
    y_test = test[target]
    # scale based on training only
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # set up and train model
    if method == "RF":
        model = RandomForestClassifier(n_estimators=500)
    elif method == "SVM":
        model = SVC(C=100, gamma=0.01, probability=True)
    else:
        model = KNeighborsClassifier(n_neighbors=20)
    model.fit(X_train, y_train.values.ravel())
    # get predictions
    predictions = model.predict(X_test)
    # compute metrics
    high_probs = model.predict_proba(X_test)[:, 0]
    brier = brier_score_loss(y_test, high_probs, pos_label="HIGH")
    metrics = pd.DataFrame(classification_report(y_test, predictions, output_dict=True))
    metrics["Brier Score"] = [brier for _ in range(4)]
    metrics.to_csv("..\\Results\\ML_results\\Test_set\\" + method + "_classification_metrics.csv")
    return predictions, high_probs


train_data = pd.read_csv("..\\Data\\MOF_data.csv")
test_data = pd.read_csv("..\\Data\\MOF_data_test.csv")

# regression
regression_metrics = []
target_predictions = pd.DataFrame()
for target in regression_targets:
    r_metrics, t_preds = test_model_regression(train_data, test_data, final_descriptors, target,
                                               SVM_parameters[target]["C"], SVM_parameters[target]["epsilon"],
                                               SVM_parameters[target]["gamma"])
    for met in r_metrics:
        regression_metrics.append(met)
    t_preds = pd.DataFrame(data=np.array(t_preds).T, columns=["MOF", "Target", "Real", "MLR", "SVM", "RF"])
    target_predictions = pd.concat([target_predictions, t_preds])
regression_metrics = pd.DataFrame(data=regression_metrics, columns=["Target", "Method", "MAE", "R2"])
regression_metrics.to_csv("../Results/ML_results/Test_set/regression_metrics.csv")
target_predictions.to_csv("../Results/ML_results/Test_set/regression_predictions.csv")

# classification
# ML_methods = ["RF", "SVM", "KNN"]
# target = "TSN Class"
# for method in ML_methods:
#     predictions, probs = test_model_classification(train_data, test_data, final_descriptors, target, method)
#     predictions = pd.DataFrame(data=np.array([test_data["MOF"], test_data[target], predictions, probs]).T,
#                                columns=["MOF", target, target + " Prediction", "HIGH Probability"])
#     predictions.to_csv("..\\Results\\ML_results\\Test_set\\" + method + "_classification_predictions.csv")
#     fpr, tpr, thresholds = roc_curve(test_data["TSN Class"], probs, pos_label="HIGH")
#     # Evaluating model performance at various thresholds
#     df_roc = pd.DataFrame(
#         {
#             'False Positive Rate': fpr,
#             'True Positive Rate': tpr
#         }, index=thresholds)
#     df_roc.index.name = "Thresholds"
#     df_roc.columns.name = "Rate"
#     df_roc.to_csv("..\\Results\\ML_results\\Test_set\\" + method + "_roc.csv")
