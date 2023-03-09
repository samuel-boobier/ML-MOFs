import numpy as np
import pandas as pd

from featureSelection import get_all_feature_subsets
from models import test_model, get_model
from preprocessing import get_data
from combiner import get_table_test

def scaler_vs_feature_selection(n, model, name, dataset):
    
    scaler_name = {'Scaler': ['Unscaled',
                              'Standard',
                              'Min-max',
                              'Max-abs',
                              'Robust',
                              'Quantile (uniform)',
                              'Quantile (gaussian)']}
    scaler_name = pd.DataFrame(scaler_name)
    
    variance_mae = np.zeros(shape=(7, 1))
    pearson_mae = np.zeros(shape=(7, 1))
    spearman_mae = np.zeros(shape=(7, 1))
    vip_mae = np.zeros(shape=(7, 1))
    lasso_mae = np.zeros(shape=(7, 1))
    mrmr_mae = np.zeros(shape=(7, 1))
    ridge_mae = np.zeros(shape=(7, 1))
    
    variance_screen = np.zeros(shape=(7, 1))
    pearson_screen = np.zeros(shape=(7, 1))
    spearman_screen = np.zeros(shape=(7, 1))
    vip_screen = np.zeros(shape=(7, 1))
    lasso_screen = np.zeros(shape=(7, 1))
    mrmr_screen = np.zeros(shape=(7, 1))
    ridge_screen = np.zeros(shape=(7, 1))
    
    
    for scaler in range(0, len(scaler_name)):
        
        train_df, test_df, target_errors = get_data(name=name, dataset=dataset, transformation=scaler)
        selectors = get_all_feature_subsets(train_df)
        
        print("Variance Model train")
        _, variance_model = get_model(model, train_df, selectors[0][1][0:n])
        print("Pearson's Model train")
        _, pearson_model = get_model(model, train_df, selectors[1][1][0:n])
        print("Spearman's Model train")
        _, spearman_model = get_model(model, train_df, selectors[2][1][0:n])
        print("VIP Model train")
        _, vip_model = get_model(model, train_df, selectors[3][1][0:n])
        print("Lasso Model train")
        _, lasso_model = get_model(model, train_df, selectors[4][1][0:n])
        print("mRMR Model train")
        _, mrmr_model = get_model(model, train_df, selectors[5][1][0:n])
        print("Ridge Model train")
        _, ridge_model = get_model(model, train_df, selectors[6][1][0:n])
        
        print("Variance Model test")
        variance_temp = get_table_test(test_df, variance_model, selectors[0][1][0:n], target_errors)
        variance_mae[scaler] = variance_temp['MAE']
        variance_screen[scaler] = variance_temp['Screen']
        print("Pearson's Model test")
        pearson_temp = get_table_test(test_df, pearson_model, selectors[1][1][0:n], target_errors)
        pearson_mae[scaler] = pearson_temp['MAE']
        pearson_screen[scaler] = pearson_temp['Screen']
        print("Spearman's Model test")
        spearman_temp= get_table_test(test_df, spearman_model, selectors[2][1][0:n], target_errors)
        spearman_mae[scaler] = spearman_temp['MAE']
        spearman_screen[scaler] = spearman_temp['Screen']
        print("VIP Model test")
        vip_temp = get_table_test(test_df, vip_model, selectors[3][1][0:n], target_errors)
        vip_mae[scaler] = vip_temp['MAE']
        vip_screen[scaler] = vip_temp['Screen']
        print("Lasso Model test")
        lasso_temp = get_table_test(test_df, lasso_model, selectors[4][1][0:n], target_errors)
        lasso_mae[scaler] = lasso_temp['MAE']
        lasso_screen[scaler] = lasso_temp['Screen']
        print("mRMR Model test")
        mrmr_temp = get_table_test(test_df, mrmr_model, selectors[5][1][0:n], target_errors)
        mrmr_mae[scaler] = mrmr_temp['MAE']
        mrmr_screen[scaler] = mrmr_temp['Screen']
        print("Ridge Model test")
        ridge_temp = get_table_test(test_df, ridge_model, selectors[6][1][0:n], target_errors)
        ridge_mae[scaler] = ridge_temp['MAE']
        ridge_screen[scaler] = ridge_temp['Screen']
    
    results_mae = pd.DataFrame(data = np.hstack([variance_mae,
                                             pearson_mae,
                                             spearman_mae,
                                             vip_mae,
                                             lasso_mae,
                                             mrmr_mae,
                                             ridge_mae]),
                           
                           index = ['Unscaled',
                                    'Standard',
                                    'Min-max',
                                    'Max-abs',
                                    'Robust',
                                    'Quantile (uniform)',
                                    'Quantile (gaussian)'],
                           
                           columns = ['Variance Threshold',
                                      "Pearson's",
                                      "Spearman's",
                                      "VIP",
                                      "LASSO",
                                      "mRMR",
                                      "Ridge"])
    
    results_screen = pd.DataFrame(data = np.hstack([variance_screen,
                                             pearson_screen,
                                             spearman_screen,
                                             vip_screen,
                                             lasso_screen,
                                             mrmr_screen,
                                             ridge_screen]),
                           
                           index = ['Unscaled',
                                    'Standard',
                                    'Min-max',
                                    'Max-abs',
                                    'Robust',
                                    'Quantile (uniform)',
                                    'Quantile (gaussian)'],
                           
                           columns = ['Variance Threshold',
                                      "Pearson's",
                                      "Spearman's",
                                      "VIP",
                                      "LASSO",
                                      "mRMR",
                                      "Ridge"])
    
    mincolval_mae = results_mae.min()
    minrowval_mae = results_mae.min(axis=1)
    
    min_combo_mae = (mincolval_mae.idxmin(), minrowval_mae.idxmin())
    min_value_mae = mincolval_mae.min()
    
    maxcolval_screen = results_screen.max()
    maxrowval_screen = results_screen.max(axis=1)
    
    max_combo_screen = (maxcolval_screen.idxmax(), maxrowval_screen.idxmax())
    max_value_screen = maxcolval_screen.max()
    
    return results_mae, min_combo_mae, min_value_mae, results_screen, max_combo_screen, max_value_screen

def add_models(name, dataset, n_features):
    
    #rf_tab_mae, rf_min_combo_mae, rf_min_mae, rf_tab_screen, rf_max_combo_screen, rf_max_screen = scaler_vs_feature_selection(n_features, 1, name, dataset)
    svrP_tab_mae, svrP_min_combo_mae, svrP_min_mae, svrP_tab_screen, svrP_max_combo_screen, svrP_max_screen = scaler_vs_feature_selection(n_features, 4, name, dataset)
    mlr_tab_mae, mlr_min_combo_mae, mlr_min_mae, mlr_tab_screen, mlr_max_combo_screen, mlr_max_screen = scaler_vs_feature_selection(n_features, 13, name, dataset)
    gpr_tab_mae, gpr_min_combo_mae, gpr_min_mae, gpr_tab_screen, gpr_max_combo_screen, gpr_max_screen = scaler_vs_feature_selection(n_features, 14, name, dataset)
    ridge_tab_mae, ridge_min_combo_mae, ridge_min_mae, ridge_tab_screen, ridge_max_combo_screen, ridge_max_screen = scaler_vs_feature_selection(n_features, 16, name, dataset)
    
    mae_mins = pd.DataFrame(data = np.hstack([#rf_min_mae,
                                          svrP_min_mae,
                                          #sgdSE_min_mae,
                                          mlr_min_mae,
                                          gpr_min_mae,
                                          ridge_min_mae]),
                        
                        index = [#"Random Forest",
                                 "SVR (Polynomial)",
                                 #"Stochastic Gradient Descent (Squared-Epsilon)",
                                 "Multiple Linear Regression",
                                 "Gaussian Process Regression",
                                 "Ridge Regression"],
                        
                        columns = ["Best MAE Score"])
    
    best_mae, best_model_mae = mae_mins.min(), mae_mins.idxmin()
    pd.to_numeric(best_mae)
    
    screen_maxes = pd.DataFrame(data = np.hstack([#rf_min_mae,
                                          svrP_min_mae,
                                          #sgdSE_min_mae,
                                          mlr_min_mae,
                                          gpr_min_mae,
                                          ridge_max_screen]),
                        
                        index = [#"Random Forest",
                                 "SVR (Polynomial)",
                                 #"Stochastic Gradient Descent (Squared-Epsilon)",
                                 "Multiple Linear Regression",
                                 "Gaussian Process Regression",
                                 "Ridge Regression"],
                        
                        columns = ["Best Screen"])
    
    best_screen, best_model_screen = screen_maxes.max(), screen_maxes.idxmax()
    pd.to_numeric(best_screen)
    
    model_tables = [
        ("SVR Polynomial", svrP_tab_mae, svrP_min_combo_mae, svrP_min_mae, svrP_tab_screen, svrP_max_combo_screen, svrP_max_screen),
        ("Multiple Linear Regression", mlr_tab_mae, mlr_min_combo_mae, mlr_min_mae, mlr_tab_screen, mlr_max_combo_screen, mlr_max_screen),
        ("Gaussian Process Regression", gpr_tab_mae, gpr_min_combo_mae, gpr_min_mae, gpr_tab_screen, gpr_max_combo_screen, gpr_max_screen),
        ("Ridge Regression", ridge_tab_mae, ridge_min_combo_mae, ridge_min_mae, ridge_tab_screen, ridge_max_combo_screen, ridge_max_screen)
        ]
    
    return model_tables, best_mae[0], str(best_model_mae[0]), best_screen[0], str(best_model_screen[0])
    

def add_feature_subsets(name, dataset):
    
    model_comp_4_features, best_mae_4, best_model_mae_4, best_screen_4, best_model_screen_4 = add_models(name, dataset, 4)
    model_comp_5_features, best_mae_5, best_model_mae_5, best_screen_5, best_model_screen_5 = add_models(name, dataset, 5)
    model_comp_6_features, best_mae_6, best_model_mae_6, best_screen_6, best_model_screen_6 = add_models(name, dataset, 6)
    model_comp_7_features, best_mae_7, best_model_mae_7, best_screen_7, best_model_screen_7 = add_models(name, dataset, 7)
    model_comp_8_features, best_mae_8, best_model_mae_8, best_screen_8, best_model_screen_8 = add_models(name, dataset, 8)
    model_comp_9_features, best_mae_9, best_model_mae_9, best_screen_9, best_model_screen_9 = add_models(name, dataset, 9)
    model_comp_10_features, best_mae_10, best_model_mae_10, best_screen_10, best_model_screen_10 = add_models(name, dataset, 10)
    model_comp_11_features, best_mae_11, best_model_mae_11, best_screen_11, best_model_screen_11 = add_models(name, dataset, 11)
    model_comp_12_features, best_mae_12, best_model_mae_12, best_screen_12, best_model_screen_12 = add_models(name, dataset, 12)
    model_comp_13_features, best_mae_13, best_model_mae_13, best_screen_13, best_model_screen_13 = add_models(name, dataset, 13)
    model_comp_14_features, best_mae_14, best_model_mae_14, best_screen_14, best_model_screen_14 = add_models(name, dataset, 14)
    model_comp_15_features, best_mae_15, best_model_mae_15, best_screen_15, best_model_screen_15 = add_models(name, dataset, 15)
    model_comp_16_features, best_mae_16, best_model_mae_16, best_screen_16, best_model_screen_16 = add_models(name, dataset, 16)
    model_comp_17_features, best_mae_17, best_model_mae_17, best_screen_17, best_model_screen_17 = add_models(name, dataset, 17)
    model_comp_18_features, best_mae_18, best_model_mae_18, best_screen_18, best_model_screen_18 = add_models(name, dataset, 18)
    model_comp_19_features, best_mae_19, best_model_mae_19, best_screen_19, best_model_screen_19 = add_models(name, dataset, 19)
    model_comp_20_features, best_mae_20, best_model_mae_20, best_screen_20, best_model_screen_20 = add_models(name, dataset, 20)
    model_comp_21_features, best_mae_21, best_model_mae_21, best_screen_21, best_model_screen_21 = add_models(name, dataset, 21)
    
    grandmaster_table = [
        ("4 Features", model_comp_4_features, best_mae_4, best_model_mae_4, best_screen_4, best_model_screen_4),
        ("5 Features", model_comp_5_features, best_mae_5, best_model_mae_5, best_screen_5, best_model_screen_5),
        ("6 Features", model_comp_6_features, best_mae_6, best_model_mae_6, best_screen_6, best_model_screen_6),
        ("7 Features", model_comp_7_features, best_mae_7, best_model_mae_7, best_screen_7, best_model_screen_7),
        ("8 Features", model_comp_8_features, best_mae_8, best_model_mae_8, best_screen_8, best_model_screen_8),
        ("9 Features", model_comp_9_features, best_mae_9, best_model_mae_9, best_screen_9, best_model_screen_9),
        ("10 Features", model_comp_10_features, best_mae_10, best_model_mae_10, best_screen_10, best_model_screen_10),
        ("11 Features", model_comp_11_features, best_mae_11, best_model_mae_11, best_screen_11, best_model_screen_11),
        ("12 Features", model_comp_12_features, best_mae_12, best_model_mae_12, best_screen_12, best_model_screen_12),
        ("13 Features", model_comp_13_features, best_mae_13, best_model_mae_13, best_screen_13, best_model_screen_13),
        ("14 Features", model_comp_14_features, best_mae_14, best_model_mae_14, best_screen_14, best_model_screen_14),
        ("15 Features", model_comp_15_features, best_mae_15, best_model_mae_15, best_screen_15, best_model_screen_15),
        ("16 Features", model_comp_16_features, best_mae_16, best_model_mae_16, best_screen_16, best_model_screen_16),
        ("17 Features", model_comp_17_features, best_mae_17, best_model_mae_17, best_screen_17, best_model_screen_17),
        ("18 Features", model_comp_18_features, best_mae_18, best_model_mae_18, best_screen_18, best_model_screen_18),
        ("19 Features", model_comp_19_features, best_mae_19, best_model_mae_19, best_screen_19, best_model_screen_19),
        ("20 Features", model_comp_20_features, best_mae_20, best_model_mae_20, best_screen_20, best_model_screen_20),
        ("21 Features", model_comp_21_features, best_mae_21, best_model_mae_21, best_screen_21, best_model_screen_21)
        ]
    
    best_maes = pd.DataFrame(data = np.hstack([best_mae_4,
                                               best_mae_5,
                                               best_mae_6,
                                               best_mae_7,
                                               best_mae_8,
                                               best_mae_9,
                                               best_mae_10,
                                               best_mae_11,
                                               best_mae_12,
                                               best_mae_13,
                                               best_mae_14,
                                               best_mae_15,
                                               best_mae_16,
                                               best_mae_17,
                                               best_mae_18,
                                               best_mae_19,
                                               best_mae_20,
                                               best_mae_21,
                                               ]),
                        
                            index = ["4 Features",
                                     "5 Features",
                                     "6 Features",
                                     "7 Features",
                                     "8 Features",
                                     "9 Features",
                                     "10 Features",
                                     "11 Features",
                                     "12 Features",
                                     "13 Features",
                                     "14 Features",
                                     "15 Features",
                                     "16 Features",
                                     "17 Features",
                                     "18 Features",
                                     "19 Features",
                                     "20 Features",
                                     "21 Features",],
                        
                            columns = ["Best MAE Scores"])
    
    best_mae, best_feature_number_mae = best_maes.min(), best_maes.idxmin()
    pd.to_numeric(best_mae)
    
    best_screen = pd.DataFrame(data = np.hstack([best_screen_4,
                                               best_screen_5,
                                               best_screen_6,
                                               best_screen_7,
                                               best_screen_8,
                                               best_screen_9,
                                               best_screen_10,
                                               best_screen_11,
                                               best_screen_12,
                                               best_screen_13,
                                               best_screen_14,
                                               best_screen_15,
                                               best_screen_16,
                                               best_screen_17,
                                               best_screen_18,
                                               best_screen_19,
                                               best_screen_20,
                                               best_screen_21,
                                               ]),
                        
                            index = ["4 Features",
                                     "5 Features",
                                     "6 Features",
                                     "7 Features",
                                     "8 Features",
                                     "9 Features",
                                     "10 Features",
                                     "11 Features",
                                     "12 Features",
                                     "13 Features",
                                     "14 Features",
                                     "15 Features",
                                     "16 Features",
                                     "17 Features",
                                     "18 Features",
                                     "19 Features",
                                     "20 Features",
                                     "21 Features",],
                        
                            columns = ["Best Screen"])
    
    best_screen, best_feature_number_screen = best_screen.max(), best_screen.idxmax()
    pd.to_numeric(best_screen)
    
    return grandmaster_table, best_mae, best_feature_number_mae, best_screen, best_feature_number_screen

def get_better_table(name): # TAKES HOURS TO RUN
    '''
    Parameters
    ----------
    name : String
        The name of the user running the function. 
        Unique names correspond to unique filepaths.

    Returns
    -------
    supreme_table : List
        An all-encompassing dataset which provides information on all the best ML models, 
        feature numbers, data transformations, feature selector, and dataset size.
    best_mae_array : Numpy array
        An array of the best MAE values that can be used to plot the best MAE 
        over time with each dataset.

    '''
    
    table100, best_mae100, best_feature_number_mae100, best_screen100, best_feature_number_screen100 = add_feature_subsets(name, 100)
    table200, best_mae200, best_feature_number_mae200, best_screen200, best_feature_number_screen200 = add_feature_subsets(name, 200)
    table300, best_mae300, best_feature_number_mae300, best_screen300, best_feature_number_screen300 = add_feature_subsets(name, 300)
    table400, best_mae400, best_feature_number_mae400, best_screen400, best_feature_number_screen400 = add_feature_subsets(name, 400)
    table500, best_mae500, best_feature_number_mae500, best_screen500, best_feature_number_screen500 = add_feature_subsets(name, 500)
    table600, best_mae600, best_feature_number_mae600, best_screen600, best_feature_number_screen600 = add_feature_subsets(name, 600)
    table700, best_mae700, best_feature_number_mae700, best_screen700, best_feature_number_screen700 = add_feature_subsets(name, 700)
    table800, best_mae800, best_feature_number_mae800, best_screen800, best_feature_number_screen800 = add_feature_subsets(name, 800)
    table900, best_mae900, best_feature_number_mae900, best_screen900, best_feature_number_screen900 = add_feature_subsets(name, 900)
    #table1000, best_mae1000, best_feature_number_mae1000, best_screen1000, best_feature_number_screen1000 = add_feature_subsets(name, 1000)
    
    supreme_table = [
        ("100 MOFs", table100, best_mae100, best_feature_number_mae100, best_screen100, best_feature_number_screen100),
        ("200 MOFs", table200, best_mae200, best_feature_number_mae200, best_screen200, best_feature_number_screen200),
        ("300 MOFs", table300, best_mae300, best_feature_number_mae300, best_screen300, best_feature_number_screen300),
        ("400 MOFs", table400, best_mae400, best_feature_number_mae400, best_screen400, best_feature_number_screen400),
        ("500 MOFs", table500, best_mae500, best_feature_number_mae500, best_screen500, best_feature_number_screen500),
        ("600 MOFs", table600, best_mae600, best_feature_number_mae600, best_screen600, best_feature_number_screen600),
        ("700 MOFs", table700, best_mae700, best_feature_number_mae700, best_screen700, best_feature_number_screen700),
        ("800 MOFs", table800, best_mae800, best_feature_number_mae800, best_screen800, best_feature_number_screen800),
        ("900 MOFs", table900, best_mae900, best_feature_number_mae900, best_screen900, best_feature_number_screen900)
        #("1000 MOFs", table1000, best_mae1000, best_feature_number_mae1000, best_screen1000, best_feature_number_screen1000),
        ]
    
    '''
    best_mae_array = np.array([#[100, best_mae100],
                               #[200, best_mae200],
                               #[300, best_mae300],
                               #[400, best_mae400],
                               #[500, best_mae500],
                               [600, best_mae600]])
    '''
    
    return supreme_table

def join_table(main_table, new_table):
    
    new_table = [main_table, new_table]

def get_better_table_addon(name, main_table, data_size):
    '''
    Parameters
    ----------
    name : String
        The name of the user running the function. 
        Unique names correspond to unique filepaths.

    Returns
    -------
    supreme_table : List
        An all-encompassing dataset which provides information on all the best ML models, 
        feature numbers, data transformations, feature selector, and dataset size.
    best_mae_array : Numpy array
        An array of the best MAE values that can be used to plot the best MAE 
        over time with each dataset.

    '''
    
    table, best_mae, best_feature_number = add_feature_subsets(name, data_size)
    
    new_table = [("f{data_size} MOFs", table, best_mae, best_feature_number)]
    
    supreme_table = join_table(main_table, new_table)
    
    '''
    best_mae_array = np.array([#[100, best_mae100]])
    '''
    
    return supreme_table