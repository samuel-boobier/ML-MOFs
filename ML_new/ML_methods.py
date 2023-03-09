import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import smogn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression


def classification(X, y, TSN):
    # implementing train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)
    MOF_test = y_test["MOF"].tolist()
    TSN_values = y_test["TSN"].tolist()
    y_test = y_test[TSN].tolist()
    y_train = y_train[TSN].tolist()
    y = y[TSN].tolist()
    sm = SMOTE(random_state=27)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(Counter(y))
    # random forest model creation
    rfc = KNeighborsClassifier()
    rfc.fit(X_train, y_train)
    # predictions
    rfc_predict = rfc.predict(X_test)
    rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, rfc_predict))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')
    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
    plot_roc_curve(rfc, X_test, y_test)
    preds = pd.DataFrame(data=np.array([MOF_test, y_test, rfc_predict, TSN_values]).T,
                         columns=["MOF", TSN, "Predictions", "TSN"])
    return preds


def get_predictions(TSN, classification_model=False):
    # descriptors to include
    descriptors = ["PLD (log10)", "LCD (log10)", "Density (g/cc)", "VSA (m2/cc)", "VF", "DC_CH4 (log10)",
                   "DC_CO2 (log10)", "DC_H2S (log10)", "Qst_CH4", "Qst_CO2", "Qst_H2S", "Qst_H2O"]
    # get target value data
    TSN_data = pd.read_excel("../Datasets/ALLabsoluteloading_postMOSAEC.xlsx")
    # drop any missing TSN values
    TSN_data = TSN_data[["MOF", TSN, "TSN error", "TSN"]]
    print("Initial dataset size: " + str(TSN_data.shape[0]))
    TSN_data = TSN_data.dropna()
    print("After dropping missing values: " + str(TSN_data.shape[0]))
    # get dataframe with names and TSN
    descriptor_data = pd.read_excel("../Datasets/dataframe_withdims.xlsx")
    descriptor_data = descriptor_data[descriptor_data["MOF Name"].isin(TSN_data["MOF"])]
    # drop all but 2D and 3D
    descriptor_data = descriptor_data[descriptor_data["Maximum Dimensions"] > 1]
    TSN_data = TSN_data[TSN_data["MOF"].isin(descriptor_data["MOF Name"])]
    print("After dropping less than 2D: " + str(TSN_data.shape[0]))
    # sort both descriptors and target by MOF name
    descriptor_data = descriptor_data.sort_values(by="MOF Name")
    TSN_data = TSN_data.sort_values(by="MOF")
    # apply log scale to some descriptors
    descriptor_data['PLD'] = np.log10(descriptor_data.PLD.values)
    descriptor_data['LCD'] = np.log10(descriptor_data.LCD.values)
    descriptor_data['DC_CH4'] = np.log10(descriptor_data["DC_CH4"].values)
    descriptor_data['DC_CO2'] = np.log10(descriptor_data["DC_CO2"].values)
    descriptor_data['DC_H2S'] = np.log10(descriptor_data["DC_H2S"].values)
    descriptor_data = descriptor_data.rename(columns={'PLD': 'PLD (log10)'})
    descriptor_data = descriptor_data.rename(columns={'LCD': 'LCD (log10)'})
    descriptor_data = descriptor_data.rename(columns={'DC_CH4': 'DC_CH4 (log10)'})
    descriptor_data = descriptor_data.rename(columns={'DC_CO2': 'DC_CO2 (log10)'})
    descriptor_data = descriptor_data.rename(columns={'DC_H2S': 'DC_H2S (log10)'})
    # get corresponding descriptors and target value
    X = descriptor_data[descriptors]
    y = TSN_data[[TSN, "MOF", "TSN"]]
    # scale descriptors
    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(data=X, columns=descriptors)
    # # reduce dataset with smogn
    # X = X.reset_index(drop=True)
    # y = y.reset_index(drop=True)
    # all_data = pd.concat([y, X], axis="columns")
    # data_smogn = smogn.smoter(data=all_data, y=TSN)
    # print(data_smogn)
    # print("After smogn dataset size: " + str(data_smogn.shape[0]))
    # X = data_smogn[descriptors]
    # y = data_smogn[TSN]
    # classification model?
    if classification_model:
        preds = classification(X, y, TSN)
        return preds
    y = TSN_data[TSN]
    # run 10-fold cross validation
    # RF model
    model = RandomForestRegressor(n_estimators=500)
    # number of folds
    k = 10
    # set up cross validation
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    # mean absolute error list
    mae_loss = []
    # prediction list
    preds = []
    # get for each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        preds.extend(y_pred)
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)

    avg_mae_valid_loss = sum(mae_loss) / k
    print("MAE = " + str(avg_mae_valid_loss))
    print("SD = " + str(np.std(y)))
    predictions = pd.DataFrame(data=np.array([TSN_data["MOF"].tolist(), y.tolist(), preds]).T,
                               columns=["MOF", TSN, "Prediction"])
    predictions = predictions.astype({TSN: 'float', 'Prediction': 'float'})
    if TSN == "LOG10 TSN":
        predictions["Prediction"] = np.power(10, predictions["Prediction"].values)
        predictions["TSN"] = np.power(10, predictions["LOG10 TSN"].values)
    if TSN == "ROOT3 TSN":
        predictions["Prediction"] = np.power(predictions["Prediction"].values, 3)
        predictions["TSN"] = np.power(predictions["ROOT3 TSN"].values, 3)
    return predictions


def bin_data(TSN):
    TSN_data = pd.read_excel("../Datasets/ALLabsoluteloading.xlsx")
    TSN_data = TSN_data[["MOF", TSN, "TSN error"]]
    TSN_data = TSN_data.dropna()
    print(pd.cut(TSN_data[TSN], bins=10).value_counts())
    binned_df = pd.cut(TSN_data[TSN], bins=10)
    print(binned_df)
