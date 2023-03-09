#Prelim
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import KFold
from modAL.models import ActiveLearner

from preprocessing import get_data, get_remainder
from featureSelection import get_features
from models import test_model
from selectBatch import get_batch 


#Setup
def gather(username, dataset, train_df, test_df, errors, feature_subset, transformation):

    #Labeled 
    X = train_df[feature_subset].to_numpy()
    y = train_df['TSN'].to_numpy().reshape(len(train_df), 1)
    
    #Unlabeled
    X_pool, data = get_remainder(username, dataset, transformation)
    X_pool = X_pool[feature_subset].to_numpy()
    
    return X, y, feature_subset, test_df, errors, X_pool, data

def GP_regression_std(regressor, X_training, X_pool):
    ##Setup
    _, std = regressor.predict(X_pool, return_std=True)
    
    #Custom strategy 
    query_idx = get_batch(X_training=X_training,X_pool=X_pool,
                                      X_uncertainty=std,
                                      metric='euclidean')
    
    return query_idx, X_pool[query_idx]

#Execution 
def get_new(username, dataset, n_queries, train_df, test_df, errors, feature_subset, transformation, export=0):#, batch_size
    ##Load in data 
    X, y, features, test, errors, X_pool, df = gather(username, dataset, train_df, test_df, errors, feature_subset, transformation)
       
    ##Initialise Active-Learner
    kernel =  RationalQuadratic(length_scale=4.74)
    
    regressor = ActiveLearner(estimator=GaussianProcessRegressor(kernel=kernel),
                              query_strategy=GP_regression_std,
                              X_training=X, 
                              y_training=y)
    
    ##Train with Cross-Validation
    k=10
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    for train_index, test_index in kf.split(X):
        X_train, _ = X[train_index,:], X[test_index,:]
        y_train, _ = y[train_index], y[test_index]
        regressor.fit(X_train, y_train)

    results_data = test_model(model=regressor, test_data=test, 
                           features=features, errors=errors)
    
    mae = results_data['MAE']
    
    print(mae)
    
    ##Get batch by core iteration method 
    queries = []
    for idx in range(n_queries):
        #Get query + append to list
        query_idx, query_instance = regressor.query(X, X_pool)
        queries.append(query_idx)
        #Remove previously selected instance from X_pool and add it to training
        X_pool = np.delete(X_pool, query_idx, 0)
        X = np.vstack([X, query_instance])
        
    Filter_df  = df[df.index.isin(queries)]
    
    ##Save dedicated test set
    if export==1:
        Filter_df['MOF Name'].to_excel('AL_MOFs2.xlsx',index=False)
    
    else:
        pass
        
    return Filter_df
