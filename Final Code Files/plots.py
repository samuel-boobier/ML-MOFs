#------------------------------------------------------------------------------
#-----------------------------Requirements-------------------------------------
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from featureSelection import get_all_feature_subsets
from models import test_model, get_model
from preprocessing import get_data


target_name = ['TSN']
target_error = ['TSN error']
feature_names = ['PLD (log10)','LCD (log10)','Density (g/cc)',\
                     'VSA (m2/cc)','GSA (m2/g)','VF' ,'PV (cc/g)','K0_CH4',\
                         'K0_CO2','K0_H2S','K0_H2O','DC_CH4 (log10)',\
                             'DC_CO2 (log10)','DC_H2S (log10)','P_CH4',\
                                 'P_CO2','P_H2S','Qst_CH4','Qst_CO2',\
                                     'Qst_H2S','Qst_H2O']

def test_all_feature_subsets(train_df, test_df, selector, target_errors):

    selectors = get_all_feature_subsets(train_df)
    
    # initialise the arrays which will hold: feature number, mae
    rf_scores = np.zeros(shape=(21, 2))
    xgb_scores = np.zeros(shape=(21, 2))
    pls_scores = np.zeros(shape=(21, 2))
    svrL_scores = np.zeros(shape=(21, 2))
    svrP_scores = np.zeros(shape=(21, 2))
    svrR_scores = np.zeros(shape=(21, 2))
    #svrS_scores = np.zeros(shape=(21, 2))
    sgdH_scores = np.zeros(shape=(21, 2))
    #sgdSE_scores = np.zeros(shape=(21, 2))
    sgdEI_scores = np.zeros(shape=(21, 2))
    #sgdSEI_scores = np.zeros(shape=(21, 2))
    #huber_scores = np.zeros(shape=(21, 2))
    elast_scores = np.zeros(shape=(21, 2))
    mlr_scores = np.zeros(shape=(21, 2))
    
    for n in range(1, len(feature_names)+1):
        fs_name, iter_features = selectors[selector][0], selectors[selector][1][0:n]
        # runs features against all models
        
        _, pls_model = get_model(0, train_df, iter_features)
        _, rf_model = get_model(1, train_df, iter_features)
        _, xgb_model = get_model(2, train_df, iter_features)
        _, svrL_model = get_model(3, train_df, iter_features)
        _, svrP_model = get_model(4, train_df, iter_features)
        _, svrR_model = get_model(5, train_df, iter_features)
        #_, svrS_model = get_model(6, train_df, iter_features)
        _, sgdH_model = get_model(7, train_df, iter_features)
        #_, sgdSE_model = get_model(8, train_df, iter_features)
        _, sgdEI_model = get_model(9, train_df, iter_features)
        #_, sgdSEI_model = get_model(10, train_df, iter_features)
        #_, huber_model = get_model(11, train_df, iter_features)
        _, elast_model = get_model(12, train_df, iter_features)
        _, mlr_model = get_model(13, train_df, iter_features)
        
        pls_mae, _, _ = test_model(test_df, pls_model, iter_features, target_errors)
        rf_mae, _, _ = test_model(test_df, rf_model, iter_features, target_errors)
        xgb_mae, _, _ = test_model(test_df, xgb_model, iter_features, target_errors)
        svrL_mae, _, _ = test_model(test_df, svrL_model, iter_features, target_errors)
        svrP_mae, _, _ = test_model(test_df, svrP_model, iter_features, target_errors)
        svrR_mae, _, _ = test_model(test_df, svrR_model, iter_features, target_errors)
        #svrS_mae, _, _ = test_model(test_df, svrS_model, iter_features, target_errors)
        sgdH_mae, _, _ = test_model(test_df, sgdH_model, iter_features, target_errors)
        #sgdSE_mae, _, _ = test_model(test_df, sgdSE_model, iter_features, target_errors)
        sgdEI_mae, _, _ = test_model(test_df, sgdEI_model, iter_features, target_errors)
        #sgdSEI_mae, _, _ = test_model(test_df, sgdSEI_model, iter_features, target_errors)
        #huber_mae, _, _ = test_model(test_df, huber_model, iter_features, target_errors)
        elast_mae, _, _ = test_model(test_df, elast_model, iter_features, target_errors)
        mlr_mae, _, _ = test_model(test_df, mlr_model, iter_features, target_errors)
        
        pls_scores[n-1, 0] = n
        pls_scores[n-1, 1] = pls_mae
        
        rf_scores[n-1, 0] = n
        rf_scores[n-1, 1] = rf_mae
        
        xgb_scores[n-1, 0] = n
        xgb_scores[n-1, 1] = xgb_mae
        
        svrL_scores[n-1, 0] = n
        svrL_scores[n-1, 1] = svrL_mae
        
        svrP_scores[n-1, 0] = n
        svrP_scores[n-1, 1] = svrP_mae
        
        svrR_scores[n-1, 0] = n
        svrR_scores[n-1, 1] = svrR_mae
        
        #svrS_scores[n-1, 0] = n
        #svrS_scores[n-1, 1] = svrS_mae
        
        sgdH_scores[n-1, 0] = n
        sgdH_scores[n-1, 1] = sgdH_mae
        
        #sgdSE_scores[n-1, 0] = n
        #sgdSE_scores[n-1, 1] = sgdSE_mae
        
        sgdEI_scores[n-1, 0] = n
        sgdEI_scores[n-1, 1] = sgdEI_mae
        
        #sgdSEI_scores[n-1, 0] = n
        #sgdSEI_scores[n-1, 1] = sgdSEI_mae
        
        #huber_scores[n-1, 0] = n
        #huber_scores[n-1, 1] = huber_mae
        
        elast_scores[n-1, 0] = n
        elast_scores[n-1, 1] = elast_mae
        
        mlr_scores[n-1, 0] = n
        mlr_scores[n-1, 1] = mlr_mae
        
        print(n)
        
    #return fs_name, pls_scores, rf_scores, xgb_scores, svrL_scores, svrP_scores, svrR_scores, svrS_scores, sgdH_scores, sgdSE_scores, sgdEI_scores, sgdSEI_scores, elast_scores, mlr_scores
    return fs_name, pls_scores, rf_scores, xgb_scores, svrL_scores, svrP_scores, svrR_scores, sgdH_scores, sgdEI_scores, elast_scores, mlr_scores

def test_all_feature_selectors(train_df, test_df, model, target_errors): # MODIFY FOR TESTING
   
    selectors = get_all_feature_subsets(train_df)
    
    pearson_scores = np.zeros(shape=(21, 2)) # initialise the arrays which will hold: feature number, mae, rmse
    spearman_scores = np.zeros(shape=(21, 2))
    variance_scores = np.zeros(shape=(21, 2))
    mrmr_scores = np.zeros(shape=(21, 2))
    vip_scores = np.zeros(shape=(21, 2))
    lasso_scores = np.zeros(shape=(21, 2))
    ridge_scores = np.zeros(shape=(21, 2))
    
    for n in range(1, len(feature_names)+1):
        variance_features = selectors[0][1][0:n]
        pearson_features = selectors[1][1][0:n] # returns a list of features remember
        spearman_features = selectors[2][1][0:n]
        vip_features = selectors[3][1][0:n]
        lasso_features = selectors[4][1][0:n]
        mrmr_features = selectors[5][1][0:n]
        ridge_features = selectors[6][1][0:n]
        
        # runs features against model
        
        model_name, variance_model = get_model(model, train_df, variance_features)
        _, pearson_model = get_model(model, train_df, pearson_features)
        _, spearman_model = get_model(model, train_df, spearman_features)
        _, vip_model = get_model(model, train_df, vip_features)
        _, lasso_model = get_model(model, train_df, lasso_features)
        _, mrmr_model = get_model(model, train_df, mrmr_features)
        _, ridge_model = get_model(model, train_df, ridge_features)
        
        variance_mae, _, _ = test_model(test_df, variance_model, variance_features, target_errors)      
        pearson_mae, _, _ = test_model(test_df, pearson_model, pearson_features, target_errors)
        spearman_mae, _, _ = test_model(test_df, spearman_model, spearman_features, target_errors)
        vip_mae, _, _ = test_model(test_df, vip_model, vip_features, target_errors)
        lasso_mae, _, _ = test_model(test_df, lasso_model, lasso_features, target_errors)
        mrmr_mae, _, _ = test_model(test_df, mrmr_model, mrmr_features, target_errors)
        ridge_mae, _, _ = test_model(test_df, ridge_model, ridge_features, target_errors)
      
        variance_scores[n-1, 0] = n
        variance_scores[n-1, 1] = variance_mae       
        
        pearson_scores[n-1, 0] = n
        pearson_scores[n-1, 1] = pearson_mae
        
        spearman_scores[n-1, 0] = n
        spearman_scores[n-1, 1] = spearman_mae
        
        vip_scores[n-1, 0] = n
        vip_scores[n-1, 1] = vip_mae
       
        lasso_scores[n-1, 0] = n
        lasso_scores[n-1, 1] = lasso_mae
        
        mrmr_scores[n-1, 0] = n
        mrmr_scores[n-1, 1] = mrmr_mae
        
        ridge_scores[n-1, 0] = n
        ridge_scores[n-1, 1] = ridge_mae
        print(n)
        
    return model_name, pearson_scores, spearman_scores, variance_scores, mrmr_scores, vip_scores, lasso_scores, ridge_scores
    
def plot_compare_models(train_df, test_df, selector, target_errors):
    
    #fs_name, pls_scores, rf_scores, xgb_scores, svrL_scores, svrP_scores, svrR_scores, svrS_scores, sgdH_scores, sgdSE_scores, sgdEI_scores, sgdSEI_scores, huber_scores, elast_scores, mlr_scores = test_all_feature_subsets(train_df, test_df, selector, target_errors)
    fs_name, pls_scores, rf_scores, xgb_scores, svrL_scores, svrP_scores, svrR_scores, sgdH_scores, sgdEI_scores, elast_scores, mlr_scores = test_all_feature_subsets(train_df, test_df, selector, target_errors)
    
    plt.figure(figsize=(16,12), dpi=800)
    
    plt.plot(rf_scores[:, 0], rf_scores[:, 1], 'red', label='Random Forest')
    plt.plot(xgb_scores[:, 0], xgb_scores[:, 1], 'orange', label='XGradientBoost')
    plt.plot(pls_scores[:, 0], pls_scores[:, 1], 'gold', label='Partial Least Squares')
    plt.plot(svrL_scores[:, 0], svrL_scores[:, 1], 'yellow', label='SVR Linear')
    plt.plot(svrP_scores[:, 0], svrP_scores[:, 1], 'olivedrab', label='SVR Polynomial')
    plt.plot(svrR_scores[:, 0], svrR_scores[:, 1], 'forestgreen', label='SVR RBF')
    #plt.plot(svrS_scores[:, 0], svrS_scores[:, 1], 'lime', label='SVR Sigmoid')
    plt.plot(sgdH_scores[:, 0], sgdH_scores[:, 1], 'turquoise', label='SGD Huber')
    #plt.plot(sgdSE_scores[:, 0], sgdSE_scores[:, 1], 'teal', label='SGD Squared Error')
    plt.plot(sgdEI_scores[:, 0], sgdEI_scores[:, 1], 'deepskyblue', label='SGD Epsilon Insensitive')
    #plt.plot(sgdSEI_scores[:, 0], sgdSEI_scores[:, 1], 'royalblue', label='SGD Squared Epsilon Insensitive')
    #plt.plot(huber_scores[:, 0], huber_scores[:, 1], 'navy', label='Huber Regression')
    plt.plot(elast_scores[:, 0], elast_scores[:, 1], 'blueviolet', label='ElasticNet')
    plt.plot(mlr_scores[:, 0], mlr_scores[:, 1], 'violet', label='Multiple Linear Regression')
    
    plt.title(label="MAE against the all 21 features of " + fs_name + " (Unscaled) removed erratic models")
    plt.xticks(range(1, len(feature_names)+1))
    plt.xlabel(xlabel='Number of Features')
    plt.ylabel(ylabel='Mean Absolute Error')
    #plt.ylim(1.5, 10)
    
    plt.legend()
    plt.show()
    
def plot_compare_feature_selectors(train_df, test_df, model, target_errors):
    
    model_name, pearson_scores, spearman_scores, variance_scores, mrmr_scores, vip_scores, lasso_scores, ridge_scores = test_all_feature_selectors(train_df, test_df, model, target_errors)
   
    plt.figure(figsize=(10,8), dpi=500)
    
    plt.plot(pearson_scores[:, 0], pearson_scores[:, 1], 'g', label='Pearsons')
    plt.plot(spearman_scores[:, 0], spearman_scores[:, 1], 'r', label='Spearmans')
    plt.plot(variance_scores[:, 0], variance_scores[:, 1], 'b', label='Variance')
    plt.plot(mrmr_scores[:, 0], mrmr_scores[:, 1], 'y', label='mRMR')
    plt.plot(vip_scores[:, 0], vip_scores[:, 1], 'purple', label='VIP')
    plt.plot(lasso_scores[:, 0], lasso_scores[:, 1], 'pink', label='LASSO')
    plt.plot(ridge_scores[:, 0], ridge_scores[:, 1], 'orange', label='Ridge')
    
    plt.title(label='MAE of ' + model_name + ' using all feature selection methods')
    plt.xticks(range(1, len(feature_names)+1))
    plt.xlabel(xlabel='Number of Features')
    plt.ylabel(ylabel='Mean Absolute Error')
    plt.ylim(1.5, 3)
    
    plt.legend()
    plt.show()
    
def plot_best_mae_per_dataset(best_mae_array):
    
    plt.figure(figsize=(4,3), dpi=300)
    
    plt.plot(best_mae_array[:, 0], best_mae_array[:, 1])
  
    plt.title(label="Best MAE test scores across each dataset size")
    plt.xticks(range(100, 701, 100))
    plt.xlabel(xlabel='Dataset size')
    plt.ylabel(ylabel='Mean Absolute Error')
    
    plt.legend()
    plt.show()

def plot_n_features_mae(grand_table):
    
    mae_array = np.zeros((18, 2))
    
    for i in range(0, 18):
        mae_array[i, 1] = grand_table[6][1][i][2]
        mae_array[i, 0] = i+4
        
    plt.figure(figsize=(8,6), dpi=500)
    
    plt.plot(mae_array[:, 0], mae_array[:, 1], label="MAE Scores")
    
    plt.title(label="Best MAE scores for each number of features from 4 to 21")
    plt.xticks(range(4, 22, 1))
    plt.xlabel(xlabel='Number of Features')
    plt.ylabel(ylabel='Mean Absolute Error')
    
    plt.legend()
    plt.show()

def get_counts_fs_scaler(grand_table):
    
    from collections import Counter
    
    fs_array = []
    scaler_array = []
    
    for c in range(0, 18):
        for m in range(0, 8, 1):
            fs_array.append(grand_table[4][1][c][1][m][2][0])
            scaler_array.append(grand_table[4][1][c][1][m][2][1])
            
    fs_cnt = Counter(fs_array)
    sc_cnt = Counter(scaler_array)
        
    fs_name = list(fs_cnt.keys())
    fs_vals = list(fs_cnt.values())
    
    sc_name = list(sc_cnt.keys())
    sc_vals = list(sc_cnt.values())
    
    return fs_cnt, fs_name, fs_vals, sc_cnt, sc_name, sc_vals

def convert_str_int_fs(grand_table):
    
    _, fs_names, fs_vals, _, _, _ = get_counts_fs_scaler(grand_table)
    
    for fs in range(0, len(fs_names), 1):
        if fs_names[fs] == "Variance Threshold":
            fs_names[fs] = 0
        if fs_names[fs] == "Pearson's":
            fs_names[fs] = 1
        if fs_names[fs] == "Spearman's":
            fs_names[fs] = 2
        if fs_names[fs] == "VIP":
            fs_names[fs] = 3
        if fs_names[fs] == "LASSO":
            fs_names[fs] = 4
        if fs_names[fs] == "mRMR":
            fs_names[fs] = 5
        if fs_names[fs] == "Ridge":
            fs_names[fs] = 6
            
    fs_keys = np.array(fs_names)
    fs_key_counts = np.array(fs_vals)
    
    fs_array = np.column_stack((fs_keys, fs_key_counts))
    
    return fs_array
    
def get_top6_features(grand_table, username):  
    
    import pandas as pd
    
    train_df, test_df, target_errors = get_data(name=username, dataset=500, transformation=5, test_size=0.2)
    
    selectors = get_all_feature_subsets(train_df)
    
    fs_array = convert_str_int_fs(grand_table)
    
    from collections import Counter
    
    feature_list = []
    
    for i in range(0, 7, 1):
        features = selectors[fs_array[i][0]][1][0:6] * fs_array[i][1]
        feature_list += features
        
    feat_cnt = Counter(feature_list)
    
    feat_df = pd.DataFrame(
            dict(
                feat_name = list(feat_cnt.keys()),
                feat_vals = list(feat_cnt.values())
                )
            )
    
    feat_df = feat_df.sort_values('feat_vals')
    
    return feat_df

def plot_fs_scaler(grand_table, username):
    
    fs_cnt, fs_name, fs_vals, sc_cnt, sc_name, sc_vals = get_counts_fs_scaler(grand_table)
    feat_df = get_top6_features(train_df, grand_table, username)
    
    plt.figure(figsize=(10,8), dpi=600)
    plt.bar('feat_name', 'feat_vals', data=feat_df)
    plt.xticks(rotation=90)
    plt.title(label="How many times features appear in the top 6 features across all optimal models")
    plt.show()
    
    plt.figure(figsize=(10,8), dpi=600)
    plt.bar(range(len(fs_cnt)), fs_vals, tick_label=fs_name)
    plt.title(label="The count of best feature selectors across the whole SDT")
    plt.show()
    
    plt.figure(figsize=(10,8), dpi=600)
    plt.bar(range(len(sc_cnt)), sc_vals, tick_label=sc_name)
    plt.title(label="The count of best data scalers across the whole SDT")
    plt.show()

def model_scores_with_datasize(grand_table, d_size):
    
    array = np.zeros((len(grand_table[0][1]), len(grand_table[0][1][1][1])))
    
    for n_features in range(0, len(grand_table[0][1])):        
        for model in range(0, len(grand_table[0][1][1][1])):
            array[n_features, model] = grand_table[d_size][1][n_features][1][model][3]
            
    return array      
                
def find_min_model_score(grand_table, d_size):
    
    model_array = model_scores_with_datasize(grand_table, d_size)
    all_model_scores = pd.DataFrame(data = model_array,
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
                                
                                    columns = ["SVR (Polynomial)",
                                               "Ridge Regression",
                                               "Gaussian Process Regression"])
    
    best_model_score = all_model_scores.min(axis=0)
    best_model_score = pd.DataFrame(data = best_model_score)
    best_model_score = best_model_score.T
        
    return best_model_score, all_model_scores

def collate_min_scores(grand_table):
    
    best100, _ = find_min_model_score(grand_table, 0)
    best200, _ = find_min_model_score(grand_table, 1)    
    best300, _ = find_min_model_score(grand_table, 2)
    best400, _ = find_min_model_score(grand_table, 3)
    best500, _ = find_min_model_score(grand_table, 4)
    best600, _ = find_min_model_score(grand_table, 5)
    best700, _ = find_min_model_score(grand_table, 7)
    
    best_scores = pd.DataFrame(data = np.vstack([best100,
                                               best200,
                                               best300,
                                               best400,
                                               best500,
                                               best600,
                                               best700]),
                               
                               index = [100,
                                        200,
                                        300,
                                        400,
                                        500,
                                        600,
                                        700],
                               
                               columns = ["SVR (Polynomial)",
                                          "Ridge Regression",
                                          "Gaussian Process Regression"])
    
    return best_scores

def plot_models_datasize(username, grand_table, data_size):
    
    best_model_scores = collate_min_scores(grand_table)
    
    plt.figure(figsize=(10,8), dpi=500)
    
    best_model_scores.plot.line()
    
    plt.title(label="Best MAE test scores across each dataset size")
    plt.xticks(range(100, 701, 100))
    plt.xlabel(xlabel='Dataset size')
    plt.ylabel(ylabel='Mean Absolute Error')
    
    plt.legend()
    plt.show()
    
    
def plot_datasize_evolution(scores_rand, scores_actl, n_points, starting_points):
    
    plt.figure(figsize=(8,6), dpi=600)
    
    z_rand = np.polyfit(scores_rand[:, 0], scores_rand[:, 1], 2)
    z_actl = np.polyfit(scores_actl[:, 0], scores_actl[:, 1], 2)
    
    p_rand = np.poly1d(z_rand)
    p_actl = np.poly1d(z_actl)
    
    plt.plot(scores_rand[:, 0], scores_rand[:, 1], label="Random Selection")
    plt.plot(scores_actl[:, 0], scores_actl[:, 1], label="Active Learning")
    
    plt.plot(scores_rand[:, 0], p_rand(scores_rand[:, 0]))
    plt.plot(scores_actl[:, 0], p_actl(scores_actl[:, 0]))
  
    plt.title(label="Test scores with increasing training data size")
    plt.xlabel(xlabel='Dataset size')
    plt.ylabel(ylabel='Mean Absolute Error')
    
    plt.legend()
    plt.show()
    

def get_plots(username, grand_table, data_size):
    
    #plot_compare_feature_selectors(train_df, test_df, model, target_errors)
    #plot_compare_models(train_df, test_df, selector, target_errors)
    #plot_best_mae_per_dataset(best_mae_array)
    #plot_fs_scaler(grand_table=grand_table, username=username)
    #plot_n_features_mae(grand_table)
    plot_models_datasize(username, grand_table, 700)