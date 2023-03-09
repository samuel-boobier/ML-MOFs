#------------------------------------------------------------------------------
#-----------------------------Requirements-------------------------------------
#------------------------------------------------------------------------------
'''
These requirements are the modules which make up the overall
program. Ensure all files are located together.
'''

from preprocessing import get_test_predictions
from combiner import get_results_quick
from activeRegressorMain import get_new
from plots import get_plots
from plots import plot_datasize_evolution
from filteredtable import get_better_table, get_better_table_addon
from finale import run_finale


'''
Set your username by either manually programming your filepath
in the preprocessing module. Alternatively, enter your filepath
as the username and it will work the same.
'''

username = 'C:\\Users\\pczsab\\PycharmProjects\\MOF_ML-ML_Ian-Jon\\Datasets'

'''
All models:
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
'''

'''
All Feature Selectors:
    --------------
    selector : Integer
        0 - Variance
        1 - Pearson's
        2 - Spearman's
        3 - VIP
        4 - LASSO
        5 - mRMR
        6 - Ridge
'''

'''
All Data Transformations:
    -------------------
    transformation : integer
        0 - Unscaled
        1 - Standard
        2 - Min-max
        3 - Max-abs
        4 - Robust
        5 - Quantile (uniform)
        6 - Quantile (gaussian)
        7 - Sample-wise L2 Normalise
'''



#------------------------------------------------------------------------------
#--------------------------Get Quick Results-----------------------------------
#------------------------------------------------------------------------------

trained_model, Specifics, Analysis, train_df, test_df, errors, feature_subset, Test_Predictions = get_results_quick(username = username,
                                                                                                 data_size = 800,
                                                                                                 n_points = 700,
                                                                                                 random_state = True,
                                                                                                 transformation = 5,
                                                                                                 selector = 4,
                                                                                                 n = 12,
                                                                                                 model = 14,
                                                                                                 advanced = True)

Predictions = get_test_predictions(Test_Predictions)

#------------------------------------------------------------------------------
#----------------------------Active Learning-----------------------------------
#------------------------------------------------------------------------------

ActiveLearningQueries = get_new(username = username, 
                  dataset = 800, 
                  n_queries = 50, 
                  train_df = train_df, 
                  test_df = test_df, 
                  errors = errors, 
                  feature_subset = feature_subset, 
                  transformation = 5, 
                  export=0)


#------------------------------------------------------------------------------
#--------------------Random Selection vs. Active Learning----------------------
#------------------------------------------------------------------------------

'''
Here you can set the parameters for the test.

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
'''


'''
This function will return 2 arrays containing test scores.
These arrays will be put through a plotting function to visually express
the differences in random selection and active learning.
'''

'''
UNCOMMENT TO ENABLE:  
Random_Scores, ActiveLearning_Scores = run_finale(name = username, 
                                                  dataset = 900, 
                                                  starting_points = 50, 
                                                  n_points = 1, 
                                                  random = True, 
                                                  transformation = 5, 
                                                  passive_model = 4,
                                                  active_model = 14,
                                                  n_iterations = 10, 
                                                  feature_subset = ['K0_H2O',
                                                                    'K0_CO2',
                                                                    'Qst_H2O',
                                                                    'VSA (m2/cc)',
                                                                    'P_CO2','VF',
                                                                    'GSA (m2/g)',
                                                                    'Qst_CO2',
                                                                    'PV (cc/g)',
                                                                    'K0_H2S',
                                                                    'Density (g/cc)',
                                                                    'P_H2S'])

'''

#------------------------------------------------------------------------------
#-------------------------------Plots------------------------------------------
#------------------------------------------------------------------------------

'''
Working on a way to make these plots more accessible with minimal programming.
'''

#------------------------------------------------------------------------------
#---------------------------The Big Table--------------------------------------
#-----------WARNING: This takes a very long time to run (4 days+)--------------

'''
GPRvsSVRP_table, best_mae_array = get_table(username)

Full_Test_Table = get_better_table(username)
'''