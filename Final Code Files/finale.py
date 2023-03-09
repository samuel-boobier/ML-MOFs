#------------------------------------------------------------------------------
#-----------------------------Requirements-------------------------------------
#------------------------------------------------------------------------------

import pandas as pd
from preprocessing import get_train_data, get_test_data
from preprocessing import get_all_train_data
from preprocessing import get_sample_train_data, validate_n_points
from featureSelection import get_features
from models import get_model, test_model, test_model_advanced
from activeRegressor import get_new


def run_finale(name, dataset, starting_points, n_points, random, transformation, rand_model, actl_model, n_iterations, feature_subset):
    """
    Trains and tests a vast number of data entries to build a dataframe
    which can show the difference in scores between active learning
    and random selection.

    Parameters
    ----------
    name : String
        Name of user or filepath containing train/test data.
    dataset : Integer
        Specify which size dataset, 100, 200, ..., 900,...
    starting_points : Integer
        How many datapoints to initialise the trials with.
    n_points : Integer
        How many datapoints to add for each iteration.
    random : Boolean
        True for random selection, False for ordered selection.
    transformation : Integer
        Which data scaler to apply from the numbered list.
    rand_model : Integer
        Which model to train/test with in the random selection trials.
    actl_model : Integer
        Which model to train/test with in the active learning trials.
    n_iterations : Integer
        How many times to repeat each trial for an average score.
    feature_subset : List
        A list containing each feature the user would like to apply to the models.

    Returns
    -------
    mae_scores_rand : Array
        Mean Absolute Error scores for each trial in random selection.
    mae_scores_actl : Array
        Mean Absolute Error scores for each trial in active learning.

    """
    
    max_iterations = calculate_max_iterations(name, dataset, transformation, starting_points, n_points)
    
    length_full_df = calculate_length_all_df(name, dataset, transformation)
    
    test_df, train_df_rand, errors_rand, remainder_train_df_rand, remainder_errors_rand = initialise_data(name, 
                                                                                                          dataset, 
                                                                                                          starting_points, 
                                                                                                          random=False, 
                                                                                                          transformation=transformation)
    
    _, train_df_actl, errors_actl, remainder_train_df_actl, remainder_errors_actl = initialise_data(name, 
                                                                                                    dataset, 
                                                                                                    starting_points, 
                                                                                                    random=False, 
                                                                                                    transformation=transformation)
    
    result_rand = get_average_result(n_iterations, rand_model, train_df_rand, feature_subset, transformation, test_df, errors_rand)
    result_actl = get_average_result(n_iterations, actl_model, train_df_actl, feature_subset, transformation, test_df, errors_actl)
    
    starting_mae_rand = result_rand
    starting_mae_actl = result_actl
    
    mae_scores_rand = initialise_array(max_iterations)
    mae_scores_actl = initialise_array(max_iterations)
    
    mae_scores_rand[0, 0] = starting_points
    mae_scores_rand[0, 1] = starting_mae_rand
    mae_scores_actl[0, 0] = starting_points
    mae_scores_actl[0, 1] = starting_mae_actl
    
    for iteration in range(1, max_iterations):
        
        print("Iteration Number: " + str(iteration))
        
        train_df_rand, errors_rand, remainder_train_df_rand, remainder_errors_rand = get_new_rand_train_df(name, dataset, train_df_rand, remainder_train_df_rand, remainder_errors_rand, n_points, random=True, transformation=transformation, current_errors=errors_rand)
        print("Random df size = " + str(len(train_df_rand)))
        
        train_df_actl, errors_actl, remainder_train_df_actl, remainder_errors_actl = get_new_actl_train_df(train_df_actl, errors_actl, remainder_train_df_actl, remainder_errors_actl, test_df, feature_subset, length_full_df, n_points)
        print("Active Learning df size = " + str(len(train_df_actl)))
        
        result_rand = get_average_result(n_iterations, rand_model, train_df_rand, feature_subset, transformation, test_df, errors_rand)
        result_actl = get_average_result(n_iterations, actl_model, train_df_actl, feature_subset, transformation, test_df, errors_actl)
        
        mae_scores_rand[iteration, 0] = len(train_df_rand)
        mae_scores_rand[iteration, 1] = result_rand
        
        mae_scores_actl[iteration, 0] = len(train_df_actl)
        mae_scores_actl[iteration, 1] = result_actl
        
    return mae_scores_rand, mae_scores_actl
    



# Start off by getting all the test data, then choosing the amount of start data to get from which dataset.
def initialise_data(name, dataset, starting_points, random, transformation):
    
    test_df = get_test_data(transformation)
    initial_train_df, initial_errors, remainder_train_df, remainder_errors = get_train_data(name, dataset, starting_points, random, transformation)

    return test_df, initial_train_df, initial_errors, remainder_train_df, remainder_errors

# After initial data is gotten, get new data specified by a particular number of points
def get_new_rand_train_df(name, dataset, current_train_df, remainder_train_df, remainder_errors, n_points, random, transformation, current_errors):
    
    new_train_df_samples, new_error_samples, new_remainder_train_df, new_remainder_errors = get_new_rand_train_df_samples(remainder_train_df, remainder_errors, n_points, random, transformation)
    new_train_df, new_errors = join_data(current_train_df, new_train_df_samples, current_errors, new_error_samples)

    return new_train_df, new_errors, new_remainder_train_df, new_remainder_errors

def get_new_actl_train_df(current_train_df, current_errors, remainder_train_data, remainder_errors, test_df, feature_subset, length_full_df, n_points):
    
    new_train_df_samples, new_error_samples, new_remainder_train_df, new_remainder_errors = get_new_actl_train_df_samples(current_train_df, current_errors, remainder_train_data, remainder_errors, test_df, feature_subset, length_full_df, n_points)
    new_train_df, new_errors = join_data(current_train_df, new_train_df_samples, current_errors, new_error_samples)
    
    return new_train_df, new_errors, new_remainder_train_df, new_remainder_errors

# Joins the old samples with the new samples
def join_data(current_train_df, new_train_df, current_errors, new_errors):
    
    train = [current_train_df, new_train_df]
    error = [current_errors, new_errors]
    
    train_df = pd.concat(train)
    errors = pd.concat(error)
    
    return train_df, errors
   
# Retrieves new samples from the full train df which have not already been chosen 
def get_new_rand_train_df_samples(remainder_train_df, remainder_errors, n_points, random, transformation):
    
    validate_n_points(remainder_train_df, n_points)
    
    rand_train_df_samples, rand_error_samples, new_remainder_train_df, new_remainder_errors = get_sample_train_data(remainder_train_df, 
                                                                                                                  n_points, 
                                                                                                                  random, 
                                                                                                                  remainder_errors)
    
    return rand_train_df_samples, rand_error_samples, new_remainder_train_df, new_remainder_errors


#
#----- Need to return new samples, with errors, and the remaining samples from the whole dataset
#

def get_new_actl_train_df_samples(train_df, errors, remainder_train_data, remainder_errors, test_df, feature_subset, length_full_df, n_points):
    
    validate_n_points(remainder_train_data, n_points)
    
    actl_train_df_samples, actl_error_samples, new_remainder_train_df, new_remainder_errors = get_new(train_df, errors, remainder_train_data, remainder_errors, test_df, feature_subset, length_full_df, n_points)

    return actl_train_df_samples, actl_error_samples, new_remainder_train_df, new_remainder_errors










def get_average_result(n_iterations, model, train_df, feature_subset, transformation, test_df, errors):
    
    import numpy as np
    
    results = np.zeros((n_iterations, 1))
    for i in range(0, n_iterations):
        trained_model, Model_Parameters = train_model_nofs(model, 
                                                           train_df, 
                                                           feature_subset, 
                                                           transformation)
        all_results = get_all_results(test_df, 
                                  trained_model, 
                                  feature_subset, 
                                  errors)
        
        results[i] = all_results['MAE'][0]
        
    result = np.average(results)
    
    return result
    

# Train the models with a feature selector and n features
def train_model_fs(model, train_df, feature_selector, n_features, transformation):

    distributions = [
        (0,"Unscaled"),
        (1,"Standard scaling"),
        (2,"Min-max scaling"),
        (3,"Max-abs scaling"),
        (4,"Robust scaling"),
        (5,"Quantile transformation (uniform)"),
        (6,"Quantile transformation (gaussian)"),
        (7,"Sample-wise L2 normalizing")
    ]
    _, data_name = distributions[transformation]

    fs_name, feature_subset = get_features(train_df, selector=feature_selector, n=n_features)

    model_name, trained_model = get_model(model=model, training_data=train_df,
                                          features=feature_subset)
    info = {'Transformation':[data_name],'Feature Selector':[fs_name],'Model':[model_name]}
    Specifics = pd.DataFrame(info)
    
    return trained_model, Specifics

# Train the models with predetermined features
def train_model_nofs(model, train_df, feature_subset, transformation):

    distributions = [
        (0,"Unscaled"),
        (1,"Standard scaling"),
        (2,"Min-max scaling"),
        (3,"Max-abs scaling"),
        (4,"Robust scaling"),
        (5,"Quantile transformation (uniform)"),
        (6,"Quantile transformation (gaussian)"),
        (7,"Sample-wise L2 normalizing")
    ]
    _, data_name = distributions[transformation]

    model_name, trained_model = get_model(model=model, training_data=train_df,
                                          features=feature_subset)
    info = {'Transformation':[data_name],'Features':[feature_subset],'Model':[model_name]}
    Specifics = pd.DataFrame(info)
    
    return trained_model, Specifics


# --------------------------------------------------
# Get Results from models for each iteration of data
# --------------------------------------------------

def get_all_results(test_data, trained_model, features, errors):
    
    results = get_basic_results(test_data, trained_model, features, errors)
    advanced_results = get_advanced_results(test_data, trained_model, features, errors)
    
    results.update(advanced_results)
    
    return results
    

def get_basic_results(test_data, model, features, errors):
    
    results = test_model(model=model, 
                     test_data=test_data, features=features,
                     errors=errors)
    
    return results
    
def get_advanced_results(test_data, model, features, errors):
    
    advanced_results = test_model_advanced(model=model, test_data=test_data,
                                       features=features)
    
    return advanced_results

def calculate_max_iterations(name, dataset, transformation, starting_points, n_points):
    
    import math
    
    full_train_df, _ = get_all_train_data(name, dataset, transformation)
    max_iterations = math.ceil((len(full_train_df) - starting_points)/(n_points+1))-5
    
    return max_iterations

def calculate_length_all_df(name, dataset, transformation):
    
    full_df, _ = get_all_train_data(name, dataset, transformation)
    length = len(full_df)
    
    return length

def initialise_array(max_iterations):
    
    import numpy as np
    array = np.zeros((max_iterations, 2))
    
    return array
    