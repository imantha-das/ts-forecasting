# ==============================================================================
# Utility functions for train.py
# ==============================================================================
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch 
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------------------
# Time lag based features 
# ------------------------------------------------------------------------------

def windowed_dataset(seq:np.array,ws:int)->tuple:
    """
    Desc : Creates a sliding window based on the specified window size (lags). Drops last incomplete window
    Inputs :
        seq : Energy load values 
        lags : last n values of lags as specified (sliding window)
    Outputs :
        X : energy load in a windowed, i.e for 10 values - [[1,2,3],[2,3,4], ... , [6,7,8]]
        y : next value of energy load, i.e [4,5,6,...,9]
    """
    X = []
    y = []
    
    for i in range(ws + 1, len(seq) + 1):
        X.append(seq[i - (ws + 1):i - 1])
        y.append(seq[i-1])
    return np.array(X),np.array(y)

# ------------------------------------------------------------------------------
# Cyclic Features
# ------------------------------------------------------------------------------

def generate_cyclic_features(df:pd.DataFrame, col_name:str, period:int, start_num:int = 0)->pd.DataFrame:
    """
    Desc : Constructs cyclic features for given column. i.e if hour is provided there is an cyclic meaning behind between 1Am and 12Am
    Inputs :
        df : DataFrame 
        col_name : Name of column you wish to transfrom
        period : the period of values. i.e for howur its 24
        start_num : the starting number of column
    returns 
        df : DataFrame
    """

    kwargs = {
        f"sin_{col_name}" : lambda x: np.sin(2 * np.pi*(df[col_name] - start_num)/period),
        f"cos_{col_name}" : lambda x: np.cos(2 * np.pi*(df[col_name] - start_num)/period)
    }

    return df.assign(**kwargs).drop(columns = [col_name])

# ------------------------------------------------------------------------------
# Train - Validation - Test Sets
# ------------------------------------------------------------------------------

def train_val_test_split(seq:np.array, y:np.array, val_ratio:float, test_ratio:float)->tuple:
    """
    Desc : Splits dataset and target to train, validation and test sets
    Inputs :
        seq : sequence of windowed data 
        y : target columns (next value of each window) 
        val_ratio : validation size 
        test_ratio : test set size
    Outputs :
        tuple of training, validation and test set
    """
    X_train, X_test, y_train, y_test = train_test_split(seq, y, test_size = test_ratio, shuffle = False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_ratio, shuffle = False)
    return X_train, X_val, X_test, y_train, y_val, y_test

# ------------------------------------------------------------------------------
# Normalise Data
# ------------------------------------------------------------------------------
    
def normalise(X_train:np.array, X_val:np.array, X_test:np.array, y_train:np.array, y_val:np.array, y_test:np.array)->tuple:
    """
    Desc : Normalises values between -1 and 1
    Inputs
        Train-Validation-Test sets
    Outputs
        Normalised train-validation-test sets
        normaliser 
    """
    normaliser = MinMaxScaler()
    X_train_norm = normaliser.fit_transform(X_train)
    X_val_norm = normaliser.fit_transform(X_val)
    X_test_norm = normaliser.fit_transform(X_test)

    y_train_norm = normaliser.fit_transform(y_train)
    y_val_norm = normaliser.fit_transform(y_val)
    y_test_norm = normaliser.fit_transform(y_test)

    return ((X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm), normaliser)

# ------------------------------------------------------------------------------
# Tensor Dataset
# ------------------------------------------------------------------------------

def torch_dataset(X_train:np.array, X_val:np.array, X_test:np.array, y_train:np.array, y_val:np.array, y_test:np.array,batch_size:int = 64)->tuple:
    """
    Desc : PyTorch Tensor dataset function
    Inputs
        Train - Validation - Test data
    Output
        Pytorch DataLoaders for train. val and test sets
    """
    train_features = torch.Tensor(X_train)
    train_targets = torch.Tensor(y_train)
    val_features = torch.Tensor(X_val)
    val_targets = torch.Tensor(y_val)
    test_features = torch.Tensor(X_test)
    test_targets = torch.Tensor(y_test)

    train_data = TensorDataset(train_features, train_targets)
    val_data = TensorDataset(val_features, val_targets)
    test_data = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False, drop_last = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, drop_last = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, drop_last = True)

    return train_loader, val_loader, test_loader


# ==============================================================================
# Run if Main
# ==============================================================================
    
if __name__ == "__main__":
    # import data
    df = pd.read_csv("data/load_ammended.csv")

    # Generate sin_hour, cos_hour
    # Replace hour with these components as NN will inherently learn better.
    df = generate_cyclic_features(df, "hour", 24)

    # Windowed dataset - removes last incomplete window
    X,y = windowed_dataset(seq = df["energy_load"], ws=24)

    # Add time features to windowed dataset - :len(X) -> removes the respective hour DOW incomplete window 
    dow_hr = df[["day_of_week", "sin_hour", "cos_hour"]][:len(X)].values

    # Stack features
    X = np.hstack((X, dow_hr))

    # Train - Validation - Test Split 
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, 0.15, 0.15)

    # print("shapes") 
    # print(f"X_train : {X_train.shape}") #? (3124,27)
    # print(f"y_train : {y_train.shape}") #? (3124,)
    # print(f"X_val : {X_val.shape}") #? (552, 27)
    # print(f"y_val : {y_val.shape}") #? (552,)
    # print(f"X_test : {X_test.shape}") #? (649, 27)
    # print(f"y_test : {y_test.shape}") #? (649,)

    # Normalise Data
    (norm_data, normaliser) = normalise(X_train, X_val, X_test, y_train.reshape(-1,1), y_val.reshape(-1,1), y_test.reshape(-1,1))
    #? Note after normalising y has a shape of (*, 1)
    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm = norm_data
    
    # Get torch datasets
    train_loader, val_loader, test_loader = torch_dataset(
        X_train_norm, 
        X_val_norm, 
        X_test_norm, 
        y_train_norm, 
        y_val_norm, 
        y_test_norm,
        batch_size = 64
    )



