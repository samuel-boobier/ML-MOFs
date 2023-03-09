#------------------------------------------------------------------------------
#-----------------------------Requirements-------------------------------------
#------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer

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

## Functions
def user(name):
    """
    Function to set the working directory for names cached in the function.
    
    Parameters
    ----------
    name : String
        Name of the individual using the program. Accepts 'Ian', 'Jon', 'Jon Laptop', or a specific file path.
    Returns
    -------
    None.
    """
    if name == 'Ian': os.chdir('C:/Users/silve/Desktop/MOF/Data' )
    
    elif name == 'Jon': os.chdir('C:/Users/jonat/OneDrive/Documents/University/Nottingham/MOF Project/MOF_ML-ML_Ian-Jon/Datasets')
    
    elif name == 'Jon Laptop': os.chdir('C:/Users/jonat/OneDrive/Documents/University/Nottingham/MLiS Final Project/main/data/unofficial')
    else:
        os.chdir(name)

def create(dataset):
    """
    Function to create a data set by three main steps. Namely: Gather features
    and targets from data sources, Merge them, Filters out unnecessary columns,
    Logs select columns (optional), and Drops empty MOF entries. 
    Parameters
    ----------
    dataset : integer
        Can be '100', '200', '300', '400'...
        represents the particular dataset to be used.
    Returns
    -------
    data : DataFrame
        Dataframe consisting of the specified number of MOFs which contains
        MOF name, features(predictors) and target(TSN).
    """
    df_main = pd.read_csv('MOF_data.csv')
    df_labels = pd.read_excel(f'{dataset}absoluteloading.xlsx')
    # merge via join key, the key being the MOF name in this case
    df = pd.merge(left=df_main, right=df_labels, left_on='MOF Name', right_on='MOF')
    del df['MOF']
    del df['CO2 loading (mol/kg)']
    del df['CH4 loading (mol/kg)']
    del df['CO2 error (mol/kg)']
    del df['CH4 error (mol/kg)']
    del df['Selectivity (CO2)']
    del df['Selectivity error']
    
    df['PLD'] = np.log10(df.PLD.values)
    df['LCD'] = np.log10(df.LCD.values)
    df['DC_CH4'] = np.log10(df["DC_CH4"].values)
    df['DC_CO2'] = np.log10(df["DC_CO2"].values) 
    df['DC_H2S'] = np.log10(df["DC_H2S"].values)
    
    df = df.rename(columns={'VF ':'VF'})
    df = df.rename(columns={'PLD':'PLD (log10)'})
    df = df.rename(columns={'LCD':'LCD (log10)'})
    df = df.rename(columns={'DC_CH4':'DC_CH4 (log10)'})
    df = df.rename(columns={'DC_CO2':'DC_CO2 (log10)'})
    df = df.rename(columns={'DC_H2S':'DC_H2S (log10)'})
    
    data = df.dropna() 

    return data


def create_test():
    """
    Function to create a data set by three main steps. Namely: Gather features
    and targets from data sources, Merge them, Filters out unnecessary columns,
    Logs select columns (optional), and Drops empty MOF entries. 
    Parameters
    ----------
    dataset : integer
        Can be '100', '200', '300', '400'...
        represents the particular dataset to be used.
    Returns
    -------
    data : DataFrame
        Dataframe consisting of the specified number of MOFs which contains
        MOF name, features(predictors) and target(TSN).
    """
    df_main = pd.read_csv('MOF_data.csv')
    df_labels = pd.read_excel('100test1000mofs.xlsx')
    # merge via join key, the key being the MOF name in this case
    df = pd.merge(left=df_main, right=df_labels, left_on='MOF Name', right_on='MOF')
    del df['MOF']
    del df['CO2 loading (mol/kg)']
    del df['CH4 loading (mol/kg)']
    del df['CO2 error (mol/kg)']
    del df['CH4 error (mol/kg)']
    del df['Selectivity (CO2)']
    del df['Selectivity error']
    
    df['PLD'] = np.log10(df.PLD.values)
    df['LCD'] = np.log10(df.LCD.values)
    df['DC_CH4'] = np.log10(df["DC_CH4"].values)
    df['DC_CO2'] = np.log10(df["DC_CO2"].values) 
    df['DC_H2S'] = np.log10(df["DC_H2S"].values)
    
    df = df.rename(columns={'VF ':'VF'})
    df = df.rename(columns={'PLD':'PLD (log10)'})
    df = df.rename(columns={'LCD':'LCD (log10)'})
    df = df.rename(columns={'DC_CH4':'DC_CH4 (log10)'})
    df = df.rename(columns={'DC_CO2':'DC_CO2 (log10)'})
    df = df.rename(columns={'DC_H2S':'DC_H2S (log10)'})
    
    data = df.dropna() 

    return data

def get_test_predictions(predictions):
    
    Full_Test_Table = create_test()
    columns = ['MOF Name', 'TSN']
    Test_Predictions = Full_Test_Table[columns]
    Test_Predictions['TSN Predictions'] = predictions
    Test_Predictions.dropna()
    
    return Test_Predictions

def scale(df, transformation):
    """
    Applies a desired transformation to the input dataset labeled by number.
    
    Parameters
    ----------
    df : DataFrame
        Input datat to be scaled, created by the 'create' function.
    transformation : integer
        0 - Unscaled
        1 - Standard
        2 - Min-max
        3 - Max-abs
        4 - Robust
        5 - Quantile (uniform)
        6 - Quantile (gaussian)
        7 - Sample-wise L2 Normalise
        Represets the transformation you wish to apply to the dataset. 
    Returns
    -------
    scaled_data : DataFrame
        Scaled by selected transform version of the input data.
    """
    n_samples = len(df)
    X, y = df[feature_names].to_numpy(), df[target_name].to_numpy()

    distributions = [
        ("Unscaled data", X),
        ("Data after standard scaling", StandardScaler().fit_transform(X)),
        ("Data after min-max scaling", MinMaxScaler().fit_transform(X)),
        ("Data after max-abs scaling", MaxAbsScaler().fit_transform(X)),
        (
            "Data after robust scaling",
            RobustScaler(quantile_range=(25, 75)).fit_transform(X)
        ),
        (
            "Data after quantile transformation (uniform)",
            QuantileTransformer(n_quantiles=n_samples, output_distribution="uniform").fit_transform(X)
        ),
        (
            "Data after quantile transformation (gaussian)",
            QuantileTransformer(n_quantiles=n_samples, output_distribution="normal").fit_transform(X)
        ),
        ("Data after sample-wise L2 normalizing",
         Normalizer().fit_transform(X))
    ]
    

    data_name, X = distributions[transformation]
    scaled_data = pd.DataFrame(np.hstack([X, y]), 
             columns=['PLD (log10)','LCD (log10)','Density (g/cc)',
                      'VSA (m2/cc)','GSA (m2/g)','VF' ,'PV (cc/g)',
                      'K0_CH4','K0_CO2','K0_H2S','K0_H2O','DC_CH4 (log10)',
                      'DC_CO2 (log10)','DC_H2S (log10)','P_CH4','P_CO2',
                      'P_H2S','Qst_CH4','Qst_CO2','Qst_H2S','Qst_H2O',
                      'TSN'])
    print(data_name)
    
    return scaled_data

def split(scaled_data, test_size):
    """
    Splits the input data into train and test sets.
    Parameters
    ----------
    scaled_data : DataFrame
        Input data to be split, created by the scale function.
    Returns
    -------
    train_df : DataFrame
        Training set for the Ml models.
    test_df : DataFrame
        Testing set for the Ml models.
    """
    X = scaled_data[feature_names]
    y = scaled_data[target_name]
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
       
    return train_df, test_df

def clean(data, training_data, clean=0):
    """
    Function to clean training set of entires with high calculation errors.
    Parameters
    ----------
    data : DataFrame
        Gathered data.
    training_data : DataFrame
        Gathered, scaled, and split data for training.
    clean : Integer
        Defult=0 which does not clean data, 1+ only used for analysis.
    Returns
    -------
    cleaned_df, the cleaned training data provided to it
    """
    if clean == 0:
           cleaned_df = training_data

    if clean == 1:
        size = int(len(data)/8)  # 8 (Spearmans), 70 (Lasso)
        df1 = data.sort_values(by=target_error, ascending=False).head(size)
        topErrors = df1.index
        cleaned_df = training_data[~training_data.index.isin(topErrors)]

    if clean == 2:
        size = int(len(data)/70)  # 70 (Lasso), 14 (Spearman)
        df1 = data.sort_values(by=target_name, ascending=False).head(size)
        topTSN = df1.index
        cleaned_df = training_data[~training_data.index.isin(topTSN)]

    if clean == 3:
        size = int(len(data)/30)  # 8 (Spearmans), 70 (Lasso)
        df = data.sort_values(by=target_error, ascending=False).head(size)
        df1 = data.sort_values(by=target_name, ascending=False).head(size)
        df = df.index
        df1 = df1.index
        topComb = df[df.isin(df1)]
        cleaned_df = training_data[~training_data.index.isin(topComb)]

    if clean == 4:
        size = int(len(data)/8)  # just bad
        df = data.sort_values(by=target_error, ascending=False).head(size)
        df1 = data.sort_values(by=target_name, ascending=False).head(size)
        df2 = pd.concat([df, df1]).drop_duplicates()  # .reset_index(drop=True)
        topComb = df2.index
        cleaned_df = training_data[~training_data.index.isin(topComb)]

    if clean == 5:
        size = int(len(data)/100)  # just bad
        df = data.sort_values(by=target_error, ascending=False).head(size)
        df1 = data.sort_values(by=target_name, ascending=False).head(size)
        df = df.index
        df1 = df1.index
        Comb1 = df[~df.isin(df1)]
        Comb2 = df[~df1.isin(df)]

        topComb = Comb1.union(Comb2)
        cleaned_df = training_data[~training_data.index.isin(topComb)]

    return cleaned_df

# New method: Utilises a separated test set
def get_data(name, dataset, n_points, random, transformation, clean_data=0):
    """
    Function to get train and test data, using the predefined user, create,
    scale, and split functions. Note: these functions can be accessed trough 
    help().
    Parameters
    ----------
    name : String
        Name of the individual using the program.
        
    dataset : Integer
        Can be '100', '200', '300', '400'...
        represents the particular dataset to be used.
        
    n_points : Integer
        How many data points will be in the training data.
        Cannot be more than dataset.
        
    random : Boolean
        If random = True, indices will be chosen randomly for training,
        otherwise they will be ordered.
        
    transformation : integer
        Transformaiton parameter: Can be
        0 - Unscaled
        1 - Standard
        2 - Min-max
        3 - Max-abs
        4 - Robust
        5 - Quantile (uniform)
        6 - Quantile (gaussian)
        7 - Sample-wise L2 Normalise
        Represents the transformation you wish to apply to the dataset.
        
    Returns
    -------
    train_df : DataFrame
        Training set for the Ml models.
        
    test_df : DataFrame
        Testing set for the Ml models.
    """
    
    train_df, errors, _, _ = get_train_data(name, dataset, n_points, random, transformation)
    test_df = get_test_data(transformation)

    return train_df, test_df, errors

def get_train_data(name, dataset, n_points, random, transformation, clean_data=0):
    
    full_train_df, full_errors = get_all_train_data(name, dataset, transformation)
    
    idx, remainder_train_idx = generate_idx(train_df=full_train_df, n_points=n_points, random=random)
    train_df, errors = full_train_df.iloc[idx], full_errors.iloc[idx]
    remainder_train_df, remainder_errors = full_train_df.iloc[remainder_train_idx], full_errors.iloc[remainder_train_idx]

    return train_df, errors, remainder_train_df, remainder_errors

def get_test_data(transformation):
    
    testdata = create_test()
    scaled_testdata = scale(testdata, transformation)
    test_df = scaled_testdata
    
    return test_df

def get_all_train_data(name, dataset, transformation, clean_data=0):
    
    user(name)
    traindata = create(dataset) # choose largest dataset, i.e 1000
    scaled_traindata = scale(traindata, transformation)
    full_train_df = clean(data=traindata, training_data=scaled_traindata, clean=clean_data)
    
    full_errors = traindata[target_error].reset_index()
    del full_errors['index']
    
    return full_train_df, full_errors
    
def get_remainder_train_data(name, dataset, remainder_idx, transformation, clean_data=0):
    
    full_train_df, full_errors = get_all_train_data(name, dataset, transformation)
    
    remainder_train_df, remainder_errors = full_train_df.iloc[remainder_idx], full_errors.iloc[remainder_idx]
    
    return remainder_train_df, remainder_errors

def get_sample_train_data(train_df, n_points, random, errors):
    
    idx, remainder_train_idx = generate_idx(train_df=train_df, n_points=n_points, random=random)
    sample_train_df, sample_errors = train_df.iloc[idx], errors.iloc[idx]
    sample_remainder_train_df, sample_remainder_errors = train_df.iloc[remainder_train_idx], errors.iloc[remainder_train_idx]
    
    return sample_train_df, sample_errors, sample_remainder_train_df, sample_remainder_errors
    
    
# Old method: Utilises first 100 MOFs
'''
def get_data(name, dataset, transformation, clean_data=0):
    """
    Function to get train and test data, using the predefined user, create,
    scale, and split functions. Note: these functions can be accessed trough 
    help().
    Parameters
    ----------
    name : String
        Name of the individual using the program.
    dataset : integer
        Can be '100', '200', '300', '400'...
        represents the particular dataset to be used.
    transformation : integer
        Transformaiton parameter: Can be
        0 - Unscaled
        1 - Standard
        2 - Min-max
        3 - Max-abs
        4 - Robust
        5 - Quantile (uniform)
        6 - Quantile (gaussian)
        7 - Sample-wise L2 Normalise
        Represents the transformation you wish to apply to the dataset. 
    Returns
    -------
    train_df : DataFrame
        Training set for the Ml models.
    test_df : DataFrame
        Testing set for the Ml models.
    """
    user(name)
    data = create(dataset)
    scaled_data = scale(data, transformation)
    test_df = scaled_data.iloc[:100]
    train_df1 = scaled_data.iloc[100:]

    train_df = clean(data=data, training_data=train_df1, clean=clean_data)

    errors = data[target_error].reset_index()
    del errors['index']

    return train_df, test_df, errors
'''

# Al Setup
def find():
    """
    Function to create a data set of ALL mofs by three main steps. Namely: Gather features
    and targets from data sources, Merge them, Filters out unnecessary columns,
    Logs select columns (optional), and Drops empty MOF entries.
    Parameters
    ----------
    dataset : integer
        Can be '100', '200', '300', '400'...
        represents the particular dataset to be used.
    Returns
    -------
    data : DataFrame
        Dataframe consisting of the specified number of MOFs which contains
        MOF name, features(predictors) and target(TSN).
    """
    df_main = pd.read_csv('MOF_data.csv')
    df_labels = pd.read_csv('6638_cifs.csv')
    # merge via join key, the key being the MOF name in this case
    df = pd.merge(left=df_main, right=df_labels,
                  left_on='MOF Name', right_on='Name')
    del df['Name']

    df['PLD'] = np.log10(df.PLD.values)
    df['LCD'] = np.log10(df.LCD.values)
    df['DC_CH4'] = np.log10(df["DC_CH4"].values)
    df['DC_CO2'] = np.log10(df["DC_CO2"].values)
    df['DC_H2S'] = np.log10(df["DC_H2S"].values)

    df = df.rename(columns={'VF ':'VF'})
    df = df.rename(columns={'PLD':'PLD (log10)'})
    df = df.rename(columns={'LCD':'LCD (log10)'})
    df = df.rename(columns={'DC_CH4':'DC_CH4 (log10)'})
    df = df.rename(columns={'DC_CO2':'DC_CO2 (log10)'})
    df = df.rename(columns={'DC_H2S':'DC_H2S (log10)'})

    data = df.dropna()
    data['TSN'] = 0

    return data

  #Al Execute 
def get_remainder(name, dataset, transformation):
    """
    Preliminary function for active regressor module - it gathers all data
    points not used in training/testing.
    Parameters
    ----------
    name: String
        Name of the individual using the program.
    dataset: integer
        Can be '100', '200', '300', '400'...
        represents the particular dataset to be used.
    transformation: integer
    Returns
    -------
    remainder: DataFrame
        Data not used in train/test.
    all_MOFs: DataFrAME
        All vialble data points.
    """
    user(name)
    data = create(dataset) # Needed to know what index to pick up from 
    
    all_MOFs = find() # Finds all MOFs 
    all_scaled = scale(all_MOFs, transformation)
    del all_MOFs['TSN']
    del all_scaled['TSN'] # No labels so TSN isn't needed 
    remainder = all_scaled[len(data):] # What isn't used for test/train

    return remainder, all_MOFs

def generate_random_idx(train_df, n_points):
    """
    Function that chooses random numbers from a preexisting list

    Parameters
    ----------
    train_df : DataFrame
        The dataset used for training purposes.
    n_points : Integer
        The number of random points to randomly select.

    Returns
    -------
    idx : List
        A list of numbers representing the selected indices
        from the training dataset.
    new_train_list : List
        A list containing the indices that were not selected.

    """
    validate_n_points(train_df, n_points)
    
    import random
    
    train_list = list(range(1, len(train_df)))
    idx = random.sample(train_list, n_points)
    new_train_list = [e for e in train_list if e not in idx]
    
    return idx, new_train_list
    
def generate_ordered_idx(train_df, n_points):
    """
    Function that chooses ordered numbers from a preexisting list

    Parameters
    ----------
    train_df : DataFrame
        The dataset used for training purposes.
    n_points : Integer
        The number of points to select.

    Returns
    -------
    idx : List
        A list of numbers representing the selected indices from the
        training dataset.
    new_train_list : List
        A list containing the indices that were not selected.

    """
    validate_n_points(train_df, n_points)
    
    train_list = list(range(1, len(train_df)))
    idx = range(1, n_points)  
    new_train_list = [e for e in train_list if e not in idx]
    
    return idx, new_train_list

def generate_idx(train_df, n_points, random):
    """
    Combines two functions to output the final list of indices depending
    on random or ordered selection.

    Parameters
    ----------
    train_df : DataFrame
        The dataset used for training purposes.
    n_points : Integer
        The number of points to select.
    random : Boolean
        True means random, False means ordered.

    Returns
    -------
    idx : List
        A list of numbers representing the selected indices from the
        training dataset.
    new_train_list : List
        A list containing the indices that were not selected.

    """
    
    if random == True:
        idx, new_train_list = generate_random_idx(train_df, n_points)
    else:
        idx, new_train_list = generate_ordered_idx(train_df, n_points)
        
    return idx, new_train_list

def validate_n_points(train_df, n_points):
    """
    Used to determine whether the number of remaining indices in a set is
    compatible with the number chosen to select. If not enough remaining points
    error is returned.

    Parameters
    ----------
    train_df : DataFrame
        The dataset used for training purposes.
    n_points : Integer
        The number of points to select.

    Raises
    ------
    ValueError
        Not enough indices to select from.

    Returns
    -------
    None.

    """
    print("Number of points: " + str(n_points))
    print("Number of points remaining: " + str(len(train_df)))
    if n_points > len(train_df):
        raise ValueError("Number of indices cannot be larger than dataset size.")
        exit()