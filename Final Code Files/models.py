#------------------------------------------------------------------------------
#-----------------------------Requirements-------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor, HuberRegressor, ElasticNet, LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn_rvm import EMRVR 
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

target_name = ['TSN']
target_error = ['TSN error']
feature_names = ['PLD (log10)',
                 'LCD (log10)',
                 'Density (g/cc)',
                 'VSA (m2/cc)',
                 'GSA (m2/g)',
                 'VF',
                 'PV (cc/g)',
                 'K0_CH4',
                 'K0_CO2',
                 'K0_H2S',
                 'K0_H2O',
                 'DC_CH4 (log10)',
                 'DC_CO2 (log10)',
                 'DC_H2S (log10)',
                 'P_CH4',
                 'P_CO2',
                 'P_H2S',
                 'Qst_CH4',
                 'Qst_CO2',
                 'Qst_H2S',
                 'Qst_H2O']


#------------------------------------------------------------------------------
#---------------------------Model Training-------------------------------------
#------------------------------------------------------------------------------

'''
All following functions prefixed with 'train_' are built solely to fit the
model using the entered training data, while utilising 10-fold Cross Validation.
'''

def train_mlr(train_df, features):

    print('Linear Regression Training Started...')
    X = train_df[features]
    y = train_df[target_name]
    model = LinearRegression()
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('Linear Regression Training Complete')
    
    return model
    
def train_pls(train_df, features):

    print('PLS Training Started')    
    X = train_df[features]
    y = train_df[target_name]
    model = PLSRegression(n_components=2)
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('PLS Training Complete')
    
    return model
    

def train_rf(train_df, features):

    X = train_df[features]
    y = train_df[target_name]
    model = RandomForestRegressor(n_estimators=50)
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    #print(X)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    #avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    #print('Avg MAE : {}'.format(avg_mae_valid_loss))
    
    return model
    

def train_xgb(train_df, features):

    print('XGB Training Started')
    X = train_df[features]
    y = train_df[target_name]
    model = XGBRegressor()
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('XGB Training Complete')
    
    return model


def train_SVR_Linear(train_df, features):

    print('SVR Linear Training Started')
    X = train_df[features]
    y = train_df[target_name]

    pipeline = Pipeline([('model', LinearSVR(C=1.05))])
    
    model = GridSearchCV(pipeline,
                          {'model__epsilon':np.linspace(0,0.2,21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    
    
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):
        print('Iteration Started')
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print('Train test split confirmed')
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        print('Model fit on fold')
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('SVR Linear Training Complete')
    
    return model

def train_SVR_Poly(train_df, features):

    #print('SVR Poly Training Started')
    X = train_df[features]
    y = train_df[target_name]
    
    model = svm.SVR(kernel='poly', gamma='scale')
    
    '''
    pipeline = Pipeline([('model', svm.SVR(kernel='poly',C=1.05))])
    
    model = GridSearchCV(pipeline,
                          {'model__epsilon':np.linspace(0,0.5,21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    '''

    
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    #avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    #print('Avg MAE : {}'.format(avg_mae_valid_loss))
    #print('SVR Poly Training Complete')
    
    return model
    

def train_SVR_rbf(train_df, features):

    print('SVR rbf Training Started')
    X = train_df[features]
    y = train_df[target_name]    
    
    pipeline = Pipeline([('model', svm.SVR(kernel='rbf', C=1.05))])
    
    model = GridSearchCV(pipeline,
                          {'model__epsilon':np.linspace(0,0.2,21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('SVR rbf Training Complete')
    
    return model


def train_SVR_sigmoid(train_df, features):

    print('SVR sigmoid Training Started')
    X = train_df[features]
    y = train_df[target_name]    
    pipeline = Pipeline([('model', svm.SVR(kernel='sigmoid', C=1.05))])
    
    model = GridSearchCV(pipeline,
                          {'model__epsilon':np.linspace(0,0.2,21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('SVR sigmoid Training Complete')
    
    return model

def train_SGDRegressor_huber(train_df, features):

    print('SGD Huber Training Started')
    X = train_df[features]
    y = train_df[target_name]
    
    pipeline = Pipeline([('model',
                          SGDRegressor(loss='huber',learning_rate='optimal'))])
    
    model = GridSearchCV(pipeline,
                          {'model__alpha':np.linspace(0.00002,0.0001,21),
                           'model__epsilon':np.linspace(0,0.3,21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    
    
    
    
#NEED TO TRY VARYING L1 ratio - currently varying L1 ratio slows down model
        
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('SGD Huber Training Complete')
    
    
    return model

def train_SGDRegressor_squared_error(train_df, features):

    print('SGD SE Training Started')
    X = train_df[features]
    y = train_df[target_name]    
        
    pipeline = Pipeline([('model',
                          SGDRegressor(loss='squared_error',learning_rate='optimal'))])
    
    model = GridSearchCV(pipeline,
                          {'model__alpha':np.linspace(0.0001, 0.01, 21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
        
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('SGD SE Training Complete')
    
    return model

def train_SGDRegressor_squared_epsilon_insensitive(train_df, features):

    print('SGD SEI Training Started')
    X = train_df[features]
    y = train_df[target_name]
    
    pipeline = Pipeline([('model',
                          SGDRegressor(loss='squared_epsilon_insensitive',
                                       learning_rate='optimal', max_iter=4000))])
    
    model = GridSearchCV(pipeline,
                          {'model__alpha':np.linspace(0.00002,0.0001,21),
                           'model__epsilon':np.linspace(0,0.3,21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
        
    
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('SGD SEI Training Complete')
    
    return model

def train_SGDRegressor_epsilon_insensitive(train_df, features):

    print('SGD EI Training Started')
    X = train_df[features]
    y = train_df[target_name]
    
    pipeline = Pipeline([('model',SGDRegressor(loss='epsilon_insensitive',
                                               learning_rate='optimal'))])
    
    model = GridSearchCV(pipeline,
                          {'model__alpha':np.linspace(0.00002,0.0001,21),
                           'model__epsilon':np.linspace(0,0.3,21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
        
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('SGD EI Training Complete')
    
    return model

def train_HuberRegressor(train_df, features):

    print('Huber Reg Training Started')
    X = train_df[features]
    y = train_df[target_name]
    
    pipeline = Pipeline([('model',HuberRegressor(max_iter=400))])
    
    model = GridSearchCV(pipeline,
                          {'model__epsilon':np.linspace(3,6,11),
                           'model__alpha':np.linspace(0.003, 0.008,11)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
        
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('Huber Reg Training Complete')
    
    return model

def train_ElasticNet(train_df, features):

    print('ElasticNet Training Started')
    X = train_df[features]
    y = train_df[target_name]
    
    pipeline = Pipeline([('model',ElasticNet())])
    
    model = GridSearchCV(pipeline,
                          {'model__alpha':np.linspace(0.05,0.5,11),
                           'model__l1_ratio':np.linspace(0.1,0.9,11)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
        
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    print('ElasticNet Training Complete')
    
    return model

def train_ridge(train_df, features):

    X = train_df[features]
    y = train_df[target_name]
    
    
    pipeline = Pipeline([('model', Ridge())])
    
    model = GridSearchCV(pipeline,
                          {'model__alpha':np.linspace(0, 0.2, 21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    
    
    #model = Ridge()

    k=5
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    
    return model

def train_lasso(train_df, features):

    X = train_df[features]
    y = train_df[target_name]
    
    pipeline = Pipeline([('model', Lasso())])
    
    
    model = GridSearchCV(pipeline,
                          {'model__alpha':np.linspace(0, 0.2, 21)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
            
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    
    return model
##Bayesian 
def train_GPR(train_df, features):

    X = train_df[features]
    y = train_df[target_name]
    
    #kernel =  RationalQuadratic(alpha=0.145, length_scale=4.74) 1.76
    kernel =  RationalQuadratic(length_scale=4.74)    

    model = GaussianProcessRegressor(kernel=kernel) 
            
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    #print('Avg MAE : {}'.format(avg_mae_valid_loss))
    #print(model.best_params_)
    
    return model

def train_RVR_Poly(train_df, features):

    #print('SVR Poly Training Started')
    X = train_df[features]
    y = train_df[target_name]
    model = EMRVR(kernel='poly', gamma='scale')
    '''
    pipeline = Pipeline([('model', EMRVR(kernel='poly', gamma='auto'))])
    
    model = GridSearchCV(pipeline,
                          {'model__threshold_alpha':np.linspace(5e4, 5e6,3)},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    '''
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    #avg_mae_valid_loss = sum(mae_loss)/k
    
    #print('MAE of each fold - {}'.format(mae_loss))
    #print('Avg MAE : {}'.format(avg_mae_valid_loss))
    #print('SVR Poly Training Complete')
    
    return model

def train_bayesian_ridge(train_df, features):

    X = train_df[features]
    y = train_df[target_name]
    
    
    pipeline = Pipeline([('model', BayesianRidge())])
    
    model = GridSearchCV(pipeline,
                          {'model__alpha_init':[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9],
                                    'model__lambda_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-9]},
                          cv = 5, scoring="neg_mean_squared_error",verbose=0
                          )
    
    
    #model = BayesianRidge()
            
    k=5
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    
    return model

def train_ARD(train_df, features):

    X = train_df[features]
    y = train_df[target_name]
    
    model = ARDRegression(compute_score=True, n_iter=30)
      
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    mae_loss = []
    
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        
        valid_mae = mean_absolute_error(y_test, y_pred)
        mae_loss.append(valid_mae)
        
    avg_mae_valid_loss = sum(mae_loss)/k
    
    print('MAE of each fold - {}'.format(mae_loss))
    print('Avg MAE : {}'.format(avg_mae_valid_loss))
    
    return model



def get_model(model, training_data, features, cost=1):
    """
    Selects and trains a model selected by a user based on a list of ideal
    features and a training set.
    
    Parameters
    ----------
    model : Integer
        0 - Partial-Least-Squares
        1 - Random-Forest
        2 - XGBoost
        3 - Support-Vector-Regressor(Linear)
        4 - Support-Vector-Regressor(Polynomial)
        5 - Support-Vector-Regressor(Radial-basis-function)
        6 - Support-Vector-Regressor(Sigmoid)
        7 - Stochastic-Gradient-Descent(Huber)
        8 - Stochastic-Gradient-Descent(Squared-error)
        9 - Stochastic-Gradient-Descent(Epsilon-insensitive)
        10 - Stochastic-Gradient-Descent(Squared-epsilon-insensitive)
        11 - Huber-Regressor
        12 - Elastic-Net.
        13 - Multiple Linear Regression
        14 - Gaussian Process Regression
        15 - Relevance Vector Regression(Polynomial)
        16 - Ridge Regression
        17 - Bayesian Ridge Regression
        18 - LASSO Regression
        19 - ARD
        
    training_data : DataFrame
        The training set of scaled data made by the 'get_data' function.
        
    features : List
        Ideal sub-set of the total features selected by function 'get_features'
        
    Returns
    -------
    model_name : String
        Simply confirms the users choice of selected model.
        
    trained_model : ...
        Trained model ready for use.
    """
    if model == 0:
        models = [("Partial-Least-Squares", train_pls(training_data, features))]        
    if model == 1:
        models = [("Random-Forest", train_rf(training_data, features))]        
    if model == 2:
        models = [("XGBoost", train_xgb(training_data, features))]        
    if model == 3:    
        models = [("Support-Vector-Regressor(Linear)", train_SVR_Linear(training_data, features))]
    if model == 4:
        models = [("Support-Vector-Regressor(Polynomial)", train_SVR_Poly(training_data, features))]
    if model == 5:
        models = [("Support-Vector-Regressor(Radial-basis-function)", train_SVR_rbf(training_data, features))]
    if model == 6:
        models = [("Support-Vector-Regressor(Sigmoid)", train_SVR_sigmoid(training_data, features))]
    if model == 7:
        models = [("Stochastic-Gradient-Descent(Huber)", train_SGDRegressor_huber(training_data, features))]
    if model == 8:
        models = [("Stochastic-Gradient-Descent(Squared-error)", train_SGDRegressor_squared_error(training_data, features))]   
    if model == 9:
        models = [("Stochastic-Gradient-Descent(Epsilon-insensitive)", train_SGDRegressor_epsilon_insensitive(training_data, features))]
    if model == 10:
        models = [("Stochastic-Gradient-Descent(Squared-epsilon-insensitive)", train_SGDRegressor_squared_epsilon_insensitive(training_data, features))]
    if model == 11:
        models = [("Huber-Regressor", train_HuberRegressor(training_data, features))]
    if model == 12:
        models = [("Elastic-Net", train_ElasticNet(training_data, features))]
    if model == 13:
        models = [("Multiple Linear Regression", train_mlr(training_data, features))]
    if model == 14:
        models = [("GPR",train_GPR(training_data, features))]
    if model == 15:
        models = [("Relevance-Vector-Regressor(Polynomial)", train_RVR_Poly(training_data, features))]
    if model == 16:
        models = [("Ridge",train_ridge(training_data, features))]
    if model == 17:
        models = [("Bayesian Ridge",train_bayesian_ridge(training_data, features))]
    if model == 18:
        models = [("Lasso",train_lasso(training_data, features))]
    if model == 19:
        models = [("ARD",train_ARD(training_data, features))]
        
    model_name, trained_model = models[0][0], models[0][1]
    
    return model_name, trained_model


def get_all_models(model, training_data, features):
    """
    Selects and trains a model selected by a user baised on a list of ideal
    features and a training set.
    
    (RETIRED FUNCTION)

    Parameters
    ----------
    model : Integer
        0 - Partial-Least-Squares
        1 - Random-Forest
        2 - XGBoost
        3 - Support-Vector-Regressor(Linear)
        4 - Support-Vector-Regressor(Polynomial)
        5 - Support-Vector-Regressor(Radial-basis-function)
        6 - Support-Vector-Regressor(Sigmoid)
        7 - Stochastic-Gradient-Descent(Huber)
        8 - Stochastic-Gradient-Descent(Squared-error)
        9 - Stochastic-Gradient-Descent(Epsilon-insensitive)
        10 - Stochastic-Gradient-Descent(Squared-epsilon-insensitive)
        11 - Huber-Regressor
        12 - Elastic-Net.
        13 - Multiple Linear Regression
        14 - Gaussian Process Regression
        15 - Relevance Vector Regression(Polynomial)
        16 - Ridge Regression
        17 - Bayesian Ridge Regression
        18 - LASSO Regression
        19 - ARD
        
    training_data : DataFrame
        The training set of scaled data made by the 'get_data' function.
        
    features : List
        Ideal sub-set of the total features selected by function 'get_features'

    Returns
    -------
    model_name : String
        Simply confirms the users choice of selected model.
    trained_model : ...
        Trained model ready for use.
    """
    models = [
        #("Partial-Least-Squares", train_pls(training_data, features)),
        #("Random-Forest", train_rf(training_data, features)),
        #("XGBoost", train_xgb(training_data, features)),
        #("Support-Vector-Regressor(Linear)", train_SVR_Linear(training_data, features)),
        #("Support-Vector-Regressor(Polynomial)", train_SVR_Poly(training_data, features)),
        #("Support-Vector-Regressor(Radial-basis-function)", train_SVR_rbf(training_data, features)),
        #("Support-Vector-Regressor(Sigmoid)", train_SVR_sigmoid(training_data, features)),
        #("Support-Vector-Regressor(Precomputed)", train_SVR_precomp(training_data, features)),
        #("Stochastic-Gradient-Descent(Huber)", train_SGDRegressor_huber(training_data, features)),
        #("Stochastic-Gradient-Descent(Squared-error)", train_SGDRegressor_squared_error(training_data, features)),
        #("Stochastic-Gradient-Descent(Epsilon-insensitive)", train_SGDRegressor_epsilon_insensitive(training_data, features)),
        #("Stochastic-Gradient-Descent(Squared-epsilon-insensitive)", train_SGDRegressor_squared_epsilon_insensitive(training_data, features)),
        #("Huber-Regressor", train_HuberRegressor(training_data, features)),
        #("Elastic-Net", train_ElasticNet(training_data, features)),
        #("Ridge-Regression", train_ridge(training_data, features)),
        #("Lasso", train_lasso(training_data, features)),
        #("Multiple Linear Regression", train_mlr(training_data, features))

    ]       
    model_name, trained_model = models[model][0], models[model][1]
    
    return model_name, trained_model



#------------------------------------------------------------------------------
#---------------------------Model Testing--------------------------------------
#------------------------------------------------------------------------------

def test_model(test_data, model, features, errors):
    """
    Used to assess trained models perfomace on unseen test data.

    Parameters
    ----------
    test_data : DataFrame
        Unseen test data created by the 'get_data' function.
        
    trained_model : TYPE
        Model trained by 'get_model' function.
        
    features : List
        The selected n features chosen by the 'get_features' function.
        
    errors : DataFrame
        The errors in TSN caculation for all MOFs.

    Returns
    -------
    data : Dictionary
        Contains MAE, PPWTE, RMSE information
        
    """    
    X_test = test_data[features]
    y_true = test_data[target_name].reset_index()
    del y_true['index']

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=['TSN'])
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    
    #Counts how many of the predictions are in within the TSN Errors.
    y_diff = abs(y_true - y_pred)
    y_diff = y_diff.rename(columns={'TSN':'TSN error'})
    y_merged = pd.merge(left=y_true, right=errors, left_index=True,
                        right_index=True).reset_index()
    del y_merged['index']
    del y_merged['TSN']
    
    df = y_merged - y_diff
    count = df[df > 0 ].count() 
    PPWTE = round((count[0]/len(test_data))*100,1)
    
    data = {'MAE':[mae], 'RMSE':[rmse],
        'PPWTE (%)':[PPWTE]}
    
    #Counts how many 'top' MOFs are sucessfully screened - to be added
    
    return data, y_pred

def test_model_advanced(test_data, model, features):
    """
    Used to assess trained models perfomace on unseen test data.
    
    Parameters
    ----------
    test_data : DataFrame
        Unseen test data created by the 'get_data' function.
        
    trained_model : TYPE
        Model trained by 'get_model' function.
        
    features : List
        The selected n features chosen by the 'get_features' function.
        
    errors : DataFrame
        The errors in TSN caculation for all MOFs.
        
    Returns
    -------
    data : Dictionary
        Contains PHPS, Sensitivity, Specificity, PPV, and NPV information.
    """    
    X_test = test_data[features]
    y_true = test_data[target_name].reset_index()
    del y_true['index']

    y_pred  = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=['TSN'])
       
    
    #Counts how many 'top' MOFs are sucessfully screened + other STAT stuff
    y_true10u = y_true[y_true > 10 ]
    y_true10u = y_true10u.dropna()
    y_pred10u = y_pred[y_pred > 10 ]
    y_pred10u = y_pred10u.dropna()
    
    y_true10u = y_true10u.index
    y_pred10u = y_pred10u.index
    
    # Percentage of High Performing MOFs Screened out
    PHPS = np.isin(y_true10u,y_pred10u).astype(int)
    PHPSinfo = round((PHPS > 0).sum()/len(y_true10u)*100, 1)
    
    TP = np.isin(y_pred10u, y_true10u).astype(int)
    TP = round((TP > 0).sum()/len(y_pred10u)*100, 1)
    #TPinfo = str(TP) +'%'
    
    FP = np.isin(y_pred10u, y_true10u).astype(int)
    FP = round((FP < 1).sum()/len(y_pred10u)*100,1)
    #FPinfo = str(FP) +'%'
    
    y_true10d= y_true[y_true <= 10 ]
    y_true10d= y_true10d.dropna()
    y_pred10d= y_pred[y_pred <= 10 ]
    y_pred10d= y_pred10d.dropna()
    
    y_true10d= y_true10d.index
    y_pred10d= y_pred10d.index
    
    TN = np.isin(y_pred10d, y_true10d).astype(int)
    TN = round((TN > 0).sum()/len(y_pred10d)*100, 1)
    #NPinfo = str(TN) +'%'
    
    FN = np.isin(y_pred10d, y_true10d).astype(int)
    FN = round((FN < 1).sum()/len(y_pred10d)*100,1)
    #FNinfo = str(FN) +'%'
    
    Sensitivity = round(TP/(TP+FN),3)
    Specificity = round(TN/(TN+FP),3)
    Screen = Sensitivity + Specificity
    
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    
    data = {'PHPS (%)':[PHPSinfo], 'Sensitivity':[Sensitivity],
            'Specificity':[Specificity], 'Screen':[Screen], 'PPV':[PPV], 'NPV':[NPV]}
    
        
    #Advanced = pd.DataFrame(data)
    
    return data