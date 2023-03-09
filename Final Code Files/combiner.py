#------------------------------------------------------------------------------
#-----------------------------Requirements-------------------------------------
#------------------------------------------------------------------------------

import pandas as pd
from preprocessing import get_data
from featureSelection import get_features
from models import get_model, test_model, test_model_advanced


def get_result(username, data_size, n_points, random_state, transformation, selector, n, model, advanced):
    # Data
    train_df, test_df, target_errors = get_data(name=username,
                                                dataset=data_size,
                                                n_points=n_points,
                                                random=random_state,
                                                transformation=transformation)
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

    fs_name, feature_subset = get_features(train_df, selector=selector, n=n)
    
    
    
    # Train model
    model_name, trained_model = get_model(model=model, training_data=train_df,
                                          features=feature_subset)
    stuff = {'Transformation':[data_name],'Feature Selector':[fs_name],'Model':[model_name]}
    Specifics = pd.DataFrame(stuff)
    
    # Test model 
    results = test_model(model=trained_model, 
                     test_data=test_df, features=feature_subset,
                     errors=target_errors)
    
    if advanced == True:
        advanced_results = test_model_advanced(model=trained_model, test_data=test_df,
                                           features=feature_subset)
        results.update(advanced_results)

    Analysis = pd.DataFrame(results)
    
    
    try:
        f = open('testfile2.xlsx')
        print("File is already available for use")
        f.close()
    except IOError:
        test_df.to_excel('testfile2.xlsx',index=False)
    
    return trained_model, Specifics, Analysis 

def get_results_quick(username, data_size, n_points, random_state, transformation, selector, n, model, advanced):
    # Data
    train_df, test_df, target_errors = get_data(name=username,
                                                dataset=data_size,
                                                n_points=n_points,
                                                random=random_state,
                                                transformation=transformation)
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

    fs_name, feature_subset = get_features(train_df, selector=selector, n=n)
    
    
    
    # Train model
    model_name, trained_model = get_model(model=model, training_data=train_df,
                                          features=feature_subset)
    stuff = {'Transformation':[data_name],'Feature Selector':[fs_name],'Model':[model_name]}
    Specifics = pd.DataFrame(stuff)
    
    # Test model 
    results, test_predictions = test_model(model=trained_model, 
                     test_data=test_df, features=feature_subset,
                     errors=target_errors)
    
    if advanced == True:
        advanced_results = test_model_advanced(model=trained_model, test_data=test_df,
                                           features=feature_subset)
        results.update(advanced_results)

    Analysis = pd.DataFrame(results)
    
    
    try:
        f = open('testfile2.xlsx')
        print("File is already available for use")
        f.close()
    except IOError:
        test_df.to_excel('testfile2.xlsx',index=False)
    
    return trained_model, Specifics, Analysis, train_df, test_df, target_errors, feature_subset, test_predictions

def get_table_test(test_data, model, features, errors):
    
    results, _ = test_model(model=model, 
                     test_data=test_data, features=features,
                     errors=errors)
    

    advanced_results = test_model_advanced(model=model, test_data=test_data,
                                       features=features)
    results.update(advanced_results)
    
    return results