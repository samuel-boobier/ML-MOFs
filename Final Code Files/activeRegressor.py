#------------------------------------------------------------------------------
#-----------------------------Requirements-------------------------------------
#------------------------------------------------------------------------------

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import KFold
from modAL.models import ActiveLearner
from selectBatch import get_batch


def get_new(train_df, errors, remainder_train_data, remainder_errors, test_df, feature_subset, length_full_df, n_queries):
    
    X, y = split_train_data(train_df, feature_subset)

    regressor = initialise_learner(estimator = GP_regression_std, 
                                   descriptors = X, 
                                   target = y)
    
    regressor = train_cross_validation(learner = regressor, 
                                       descriptors = X, 
                                       target = y)
    
    new_inds, remaining_inds = find_queries_idx(learner = regressor,
                                               X = X,
                                               remainder_df = remainder_train_data,
                                               feature_subset = feature_subset,
                                               length_full_df = length_full_df, 
                                               n_queries = n_queries)


    
    new_train_df_samples, new_error_samples, new_remainder_train_df, new_remainder_errors = get_samples(remainder_df = remainder_train_data, 
                                                                                                        remainder_errors = remainder_errors, 
                                                                                                        new_inds = new_inds, 
                                                                                                        remaining_inds = remaining_inds)
    
    print(new_train_df_samples)


    return new_train_df_samples, new_error_samples, new_remainder_train_df, new_remainder_errors

def split_train_data(train_df, feature_subset):
    
    X = train_df[feature_subset].to_numpy() # Descriptors
    y = train_df['TSN'].to_numpy().reshape(len(train_df), 1) # Target variable
    
    return X, y

def initialise_learner(estimator, descriptors, target):

    from sklearn_rvm import EMRVR    

    
    kernel =  RationalQuadratic(length_scale=4.74)

    learner = ActiveLearner(estimator=GaussianProcessRegressor(kernel=kernel),
                          query_strategy=GP_regression_std,
                          X_training=descriptors, 
                          y_training=target)
    
    '''
    learner = ActiveLearner(estimator=EMRVR(kernel='poly', gamma='scale'),
                            query_strategy=GP_regression_std,
                            X_training=descriptors, 
                            y_training=target)
    '''

    return learner

def train_cross_validation(learner, descriptors, target, k=10):
    
    kf = KFold(n_splits=k, random_state=None, shuffle=False)
    
    for train_index, test_index in kf.split(descriptors):
        X_train, _ = descriptors[train_index,:], descriptors[test_index,:]
        y_train, _ = target[train_index], target[test_index]
        learner.fit(X_train, y_train)

    return learner

def convert_df_np_idx_retained(df, length_full_df):

    array = np.zeros((length_full_df, len(df.columns)))
    
    array[list(df.index)] = df
    
    return array

def find_queries_idx(learner, X, remainder_df, feature_subset, length_full_df, n_queries):
    
    X_pool = remainder_df[feature_subset].to_numpy()
    print(X_pool)
    queries = np.zeros((n_queries, 1))
    
    for idx in range(n_queries):

        query_idx, _ = learner.query(X, X_pool)
        queries[idx] = query_idx
        
        X_pool = np.delete(X_pool, query_idx, 0)
    
    print("Number of queries extracted: "+ str(len(queries)))
    all_inds = list(range(0, len(X_pool)))
    new_inds = [e for e in all_inds if e in queries]
    remaining_inds = [e for e in all_inds if e not in queries]
    
    return new_inds, remaining_inds

def get_samples(remainder_df, remainder_errors, new_inds, remaining_inds):
    
    remainder_df = remainder_df.reset_index(drop=True)
    remainder_errors = remainder_errors.reset_index(drop=True)    
    
    new_train_df_samples, new_error_samples = remainder_df.loc[new_inds], remainder_errors.loc[new_inds]
    new_remainder_train_df, new_remainder_errors = remainder_df.loc[remaining_inds], remainder_errors.loc[remaining_inds]
    
    return new_train_df_samples, new_error_samples, new_remainder_train_df, new_remainder_errors


def GP_regression_std(regressor, X_training, X_pool):
    ##Setup
    _, std = regressor.predict(X_pool, return_std=True)
    
    #Custom strategy 
    query_idx = get_batch(X_training=X_training,
                          X_pool=X_pool,
                          X_uncertainty=std,
                          metric='euclidean')
    
    return query_idx, X_pool[query_idx]

'''
def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]
'''