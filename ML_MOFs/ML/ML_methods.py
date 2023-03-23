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
import plotly


def prepare_data(data, descriptors, target):
    X = data[descriptors]
    y = data[[target, "MOF"]]
    # scale descriptors
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(data=X, columns=descriptors)
    return X, y


def run_model(model, X, y, target, ML_type):
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
        y_pred = model.predict(X_test)
        if ML_type == "classification":
            high_probs = model.predict_proba(X_test)[:, 0]
            low_probs = model.predict_proba(X_test)[:, 1]
            briers.append({
                "HIGH": brier_score_loss(y_test, high_probs, pos_label="HIGH"),
                "LOW": brier_score_loss(y_test, low_probs, pos_label="LOW")
            })
            y_score = model.predict_proba(X)[:, 1]
            fpr, tpr, thresholds = roc_curve(y, y_score)

            # The histogram of scores compared to true labels
            fig_hist = px.histogram(
                x=y_score, color=y, nbins=50,
                labels=dict(color='True Labels', x='Score')
            )

            fig_hist.show()

            # Evaluating model performance at various thresholds
            df = pd.DataFrame({
                'False Positive Rate': fpr,
                'True Positive Rate': tpr
            }, index=thresholds)
            df.index.name = "Thresholds"
            df.columns.name = "Rate"

            fig_thresh = px.line(
                df, title='TPR and FPR at every threshold',
                width=700, height=500
            )

            fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
            fig_thresh.show()
        preds.extend(y_pred)
        targets.extend(y_test[target].tolist())
        if ML_type == "regression":
            metric1.append(mean_absolute_error(y_test, y_pred))
            metric2.append(r2_score(y_pred, y_test))
        else:
            metric1.append(confusion_matrix(y_test, y_pred))
            metric2.append(classification_report(y_test, y_pred, output_dict=True))
    return preds, MOFS, targets, metric1, metric2, k, briers


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
    preds, MOFS, targets, mae_loss, r2, k, briers = run_model(model, X, y, target, "regression")
    # average mae
    avg_mae_valid_loss = sum(mae_loss) / k
    # average r2
    avg_r2 = sum(r2) / k
    metrics = [target, method, avg_r2, np.std(r2), avg_mae_valid_loss, np.std(mae_loss), np.std(y[target])]
    predictions = pd.DataFrame(data=np.array([MOFS, targets, preds]).T,
                               columns=["MOF", target, target + " Prediction"])
    predictions = predictions.astype({target: 'float', target + ' Prediction': 'float'})
    return predictions, metrics


def classification(data, descriptors, target, method):
    X, y = prepare_data(data, descriptors, target)
    # set up model
    if method == "RF":
        model = RandomForestClassifier(n_estimators=500)
    elif method == "KNN":
        model = KNeighborsClassifier(n_neighbors=12)
    elif method == "SVM":
        model = SVC(C=10, gamma="auto", probability=True)
    else:
        print("invalid model")
        return
    preds, MOFS, targets, confusion_matrices, classification_reports, k, briers = run_model(model, X, y, target,
                                                                                                "classification")
    predictions = pd.DataFrame(data=np.array([MOFS, targets, preds]).T,
                               columns=["MOF", target, target + " Prediction"])
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
        class_list.append(np.mean([x[c] for x in briers]))
        class_list.append(np.std([x[c] for x in briers]))
        classification_data.append(class_list)
    classification_data = pd.DataFrame(data=classification_data, columns=[
        "Class", "Precision Mean", "Precision SD", "Recall Mean", "Recall SD", "F1 Score Mean", "F1 Score SD",
        "Support Mean", "Support SD", "Accuracy Mean", "Accuracy SD", "Briers Mean", "Briers SD"
    ])
    classification_data.to_csv("..\\Results\\ML_results\\Classification\\" + method + "_classification_report.csv")
    predictions.to_csv("..\\Results\\ML_results\\Classification\\" + method + "_predictions.csv")
