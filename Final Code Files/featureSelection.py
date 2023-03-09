#------------------------------------------------------------------------------
#-----------------------------Requirements-------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


target_name = ['TSN']
target_error = ['TSN error']
feature_names = ['PLD (log10)','LCD (log10)','Density (g/cc)',\
                     'VSA (m2/cc)','GSA (m2/g)','VF' ,'PV (cc/g)','K0_CH4',\
                         'K0_CO2','K0_H2S','K0_H2O','DC_CH4 (log10)',\
                             'DC_CO2 (log10)','DC_H2S (log10)','P_CH4',\
                                 'P_CO2','P_H2S','Qst_CH4','Qst_CO2',\
                                     'Qst_H2S','Qst_H2O']

sams_feature_names = ['PLD (log10)','LCD (log10)','Density (g/cc)',\
                      'VSA (m2/cc)','VF','DC_CH4 (log10)','DC_CO2 (log10)',\
                          'DC_H2S (log10)','Qst_CH4','Qst_CO2','Qst_H2S',\
                              'Qst_H2O']
    

def sams(df):
    sams_features = sams_feature_names
    
    return sams_features
    
def mrmr(df, n):
    from pymrmre import mrmr

    X = df[feature_names]
    Y = df[target_name]
    solutions = mrmr.mrmr_ensemble(features=X, targets=Y, solution_length=21, solution_count=1)
    mrmr_features = solutions.iloc[0][0:1]
    mrmr_features = mrmr_features[0][0:n]
    
    return mrmr_features

def spearman(df, n):
    
    corr = df.corr(method='spearman')['TSN'].sort_values(ascending=False)[1:]
    abs_corr = abs(corr).sort_values(ascending=False)
    spearman_features = abs_corr[0:n]
    spearman_features = spearman_features.index.tolist()
    
    return spearman_features

def pearson(df, n):
    
    corr = df.corr()['TSN'].sort_values(ascending=False)[1:]
    abs_corr = abs(corr).sort_values(ascending=False)
    pearson_features = abs_corr[0:n]
    pearson_features = pearson_features.index.tolist()
    
    return pearson_features

def variance(df, n):

    data = df[feature_names]
    variances = data.var().sort_values(ascending=False)
    variance_features = variances[0:n]
    variance_features = variance_features.index.tolist()

    return variance_features

def vip(df, n):
    from sklearn.cross_decomposition import PLSRegression
    
    X = df[feature_names]
    y = df[target_name] 
    model = PLSRegression(n_components=2)
    model = model.fit(X, y)

    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    m, p = X.shape
    _, h = t.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    features = np.array(feature_names)
    features = pd.DataFrame(features, columns=['Feature'])
    vips = pd.DataFrame(vips, columns=['VIP'])
    vip_features = pd.concat([features, vips], axis='columns')
    vip_features = vip_features.sort_values(ascending=False, by='VIP')
    vip_features = vip_features['Feature'][0:n].tolist()
        
    return vip_features

def lasso(df,n):
    from sklearn.linear_model import Lasso
    
    X = df[feature_names]
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)
    pipeline = Pipeline([('model',Lasso())])
    
    search = GridSearchCV(pipeline,
                          {'model__alpha':np.arange(0.1,10,0.1)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    
    search.fit(X_train,y_train.values.ravel())
    search.best_params_
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    importance = pd.DataFrame(importance, columns=['Importance'])
    
    features = np.array(feature_names)
    features = pd.DataFrame(features, columns=['Feature'])
    lasso_features = pd.concat([features, importance], axis='columns')
    lasso_features = lasso_features.sort_values(ascending=False, by='Importance')
    lasso_features = lasso_features['Feature'][0:n].tolist()
    
    return lasso_features

def ridge(df,n):
    from sklearn.linear_model import Ridge
    X = df[feature_names]
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)
    pipeline = Pipeline([('model',Ridge())])
    
    search = GridSearchCV(pipeline,
                          {'model__alpha':np.arange(0.1,10,0.1)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    
    search.fit(X_train,y_train.values.ravel())
    search.best_params_
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    importance = pd.DataFrame(importance.transpose(), columns=['Importance'])
    
    features = np.array(feature_names)
    features = pd.DataFrame(feature_names, columns=['Feature'])
    ridge_features = pd.concat([features, importance], axis='columns')
    ridge_features = ridge_features.sort_values(ascending=False, by='Importance')
    ridge_features = ridge_features['Feature'][0:n].tolist()
    return ridge_features


def get_all_feature_subsets(train_df):
    """
    Gets the ordered list of 21 features from all feature selection techniques.

    Parameters
    ----------
    train_df : DataFrame
        The training set of scaled data made by the 'get_data' function.

    Returns
    -------
    selectors : Tuple
        Tuple containing each feature selection technique name as the key,
        and the ordered list of 21 features as the value.

    """
    
    selectors = [
        ("Variance", variance(train_df, 21)),
        ("Pearson's", pearson(train_df, 21)),
        ("Spearman's", spearman(train_df, 21)),
        ("VIP", vip(train_df, 21)),
        ("LASSO", lasso(train_df, 21)),
        ("mRMR", mrmr(train_df, 21)),
        ("Ridge", ridge(train_df, 21))
    ]
    
    return selectors

def get_features(train_df, selector, n):
    """
    Selects a list of features based on a specfied feature selection method.
    Parameters
    ----------
    train_df : DataFrame
        The training set of scaled data made by the 'get_data' function.
    selector : Integer
        0 - Variance
        1 - Pearson's
        2 - Spearman's
        3 - VIP
        4 - LASSO
        5 - mRMR.
        Represents the feature selection method to be applied.
        
    n : Integer
        Number of desired features.
    Returns
    -------
    fs_name : String
        Name of the feature selection method used.
    feature_subset : List
        The selected n features by the method.
    """
    selectors = [
        ("Variance", variance(train_df, 21)),
        ("Pearson's", pearson(train_df, 21)),
        ("Spearman's", spearman(train_df, 21)),
        ("VIP", vip(train_df, 21)),
        ("LASSO", lasso(train_df, 21)),
        ("mRMR", mrmr(train_df, 21)),
        ("Ridge", ridge(train_df, 21)),
        ("Sam's (PCA)", sams(train_df))
    ]       
    fs_name, feature_subset = selectors[selector][0], selectors[selector][1][0:n]
    
    return fs_name, feature_subset