import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, brier_score_loss, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression


def importance_method(importance, descriptors, target):
    importance = np.array(importance).T
    imp = [[np.mean(x), np.std(x)] for x in importance]
    for i in range(len(descriptors)):
        imp[i].insert(0, descriptors[i])
    importance = pd.DataFrame(data=imp, columns=["Descriptor", target + " Mean", target + " SD"])
    return importance


def prepare_data(data, descriptors, target):
    X = data[descriptors]
    y = data[[target, "MOF"]]
    # scale descriptors
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(data=X, columns=descriptors)
    return X, y


def run_model(model, X, y, target, ML_type, method):
    metric1 = []
    metric2 = []
    # number of folds
    k = 10
    # set up cross validation
    kf = KFold(n_splits=k, random_state=None, shuffle=True)
    # prediction list
    preds = []
    # MOFS list
    MOFS = []
    # targets list
    targets = []
    briers = []
    probs = []
    df_roc = None
    importance = []
    # get for each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # get target values
        MOFS.extend(y_test["MOF"].tolist())
        # remove target values from y
        y_train = y_train.drop(columns=["MOF"])
        y_test = y_test.drop(columns=["MOF"])
        model.fit(X_train, y_train.values.ravel())
        if method == "RF":
            importance.append(model.feature_importances_)
        y_pred = model.predict(X_test)
        if ML_type == "classification":
            high_probs = model.predict_proba(X_test)[:, 0]
            probs.extend(high_probs)
            briers.append(brier_score_loss(y_test, high_probs, pos_label="HIGH"))
        preds.extend(y_pred)
        targets.extend(y_test[target].tolist())
        if ML_type == "regression":
            metric1.append(mean_absolute_error(y_test, y_pred))
            metric2.append(r2_score(y_pred, y_test))
        else:
            metric1.append(confusion_matrix(y_test, y_pred))
            metric2.append(classification_report(y_test, y_pred, output_dict=True))
    return preds, MOFS, targets, metric1, metric2, k, briers, probs, df_roc, importance


def regression(data, descriptors, target, method, C=None, epsilon=None, gamma=None):
    X, y = prepare_data(data, descriptors, target)
    # set up model
    if method == "RF":
        model = RandomForestRegressor(n_estimators=500)
    elif method == "MLR":
        model = LinearRegression()
    elif method == "SVM":
        model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    else:
        print("invalid model")
        return
    preds, MOFS, targets, mae_loss, r2, k, briers, probs, df_roc, importance = run_model(model, X, y, target,
                                                                                         "regression", method)
    # average mae
    avg_mae_valid_loss = sum(mae_loss) / k
    # average r2
    avg_r2 = sum(r2) / k
    metrics = [target, method, avg_r2, np.std(r2), avg_mae_valid_loss, np.std(mae_loss), np.std(y[target])]
    predictions = pd.DataFrame(data=np.array([MOFS, targets, preds]).T,
                               columns=["MOF", target, target + " Prediction"])
    predictions = predictions.astype({target: 'float', target + ' Prediction': 'float'})
    if method == "RF":
        importance = importance_method(importance, descriptors, target)
    else:
        importance = None
    return predictions, metrics, importance


def classification(data, descriptors, target, method):
    X, y = prepare_data(data, descriptors, target)
    # set up model
    if method == "RF":
        model = RandomForestClassifier(n_estimators=500)
    elif method == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif method == "SVM":
        model = SVC(C=1, gamma="scale", probability=True)
    else:
        print("invalid model")
        return
    preds, MOFS, targets, confusion_matrices, classification_reports, \
    k, briers, probs, df_roc, importance = run_model(model, X, y, target, "classification", method)
    predictions = pd.DataFrame(data=np.array([MOFS, targets, preds, probs]).T,
                               columns=["MOF", target, target + " Prediction", "HIGH Probability"])
    classes = ["HIGH", "LOW"]
    metrics = ["precision", "recall", "f1-score", "support"]
    classification_data = []
    for c in classes:
        class_list = [c]
        for m in metrics:
            class_list.append(np.mean([x[c][m] for x in classification_reports]))
            class_list.append(np.std([x[c][m] for x in classification_reports]))
        class_list.append(np.mean([x["accuracy"] for x in classification_reports]))
        class_list.append(np.std([x["accuracy"] for x in classification_reports]))
        class_list.append(np.mean(briers))
        class_list.append(np.std(briers))
        classification_data.append(class_list)
    classification_data = pd.DataFrame(data=classification_data, columns=[
        "Class", "Precision Mean", "Precision SD", "Recall Mean", "Recall SD", "F1 Score Mean", "F1 Score SD",
        "Support Mean", "Support SD", "Accuracy Mean", "Accuracy SD", "Briers Mean", "Briers SD"
    ])
    classification_data.to_csv("..\\Results\\ML_results\\Classification\\" + method + "_classification_report.csv")
    predictions.to_csv("..\\Results\\ML_results\\Classification\\" + method + "_predictions.csv")
    fpr, tpr, thresholds = roc_curve(targets, probs, pos_label="HIGH")
    # Evaluating model performance at various thresholds
    df_roc = pd.DataFrame(
        {
            'False Positive Rate': fpr,
            'True Positive Rate': tpr
        }, index=thresholds)
    df_roc.index.name = "Thresholds"
    df_roc.columns.name = "Rate"
    df_roc.to_csv("..\\Results\\ML_results\\Classification\\" + method + "_roc.csv")
    if method == "RF":
        importance = importance_method(importance, descriptors, target)
        importance.to_csv("..\\Results\\ML_results\\Classification\\" + method + "_importance.csv")
