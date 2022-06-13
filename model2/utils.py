import os
import numpy as np 
import pandas as pd

from dateutil.relativedelta import relativedelta

import torch

import plotly.graph_objs as go

def load_data(path, last_date = "2022-05-15 23:00:00", endpoint=31):
    """
    Loads dataset
    Inputs
        - path : Files path to xslx (str)
        - last date : latst date and time to populate dates (str)
        - endpoint : Remove time series that donot have entire days worth of data (int)
    Outputs
        - df : pandas dataframe (pd.DataFrame)
    """
    df = pd.read_excel(path, header = None)
    df = df[:-endpoint]
    df.rename(columns = {0 : "y"}, inplace = True)
    last_ts = pd.to_datetime(last_date, format = "%Y-%m-%d %H:%M:%S")
    ts = [last_ts - relativedelta(hours=  x) for x in range(df.shape[0])]
    ts.sort()
    df["date"] = ts
    return df 
    
def windowed_dataset(seq, ws):
    """
    Creates features from a sequence of size 
    Inputs 
        seq : timeseries
        ws : window size
    Outputs
        X : sequence split into time series
        y : next value in sequence 
    """
    X = []
    y = []
    for i in range(ws + 1, len(seq) + 1):
        X.append(seq[i - (ws + 1):i - 1])
        y.append(seq[i-1])
    return X, y


def get_input_data(seq,ws,split_at,train_size = 0.7):
    """
    Splits dataset to train, validation and test
    Inputs :
        seq : timeseries
        split_at : index to split train and test set
        train_size : percentage of data to split into train and validation
    Outputs :
        X_train : feature training set
        y_train : target training set
        X_val : feature validation set
        y_val : target validation set 
        X_test : feature testing set
        y_test : target testing set
    """
    X, y = windowed_dataset(seq, ws)
    X_train, y_train, X_test, y_test = X[:split_at],y[:split_at], X[split_at:], y[split_at:]
    train_len = round(len(X_train) * train_size)
    X_train, y_train, X_val, y_val = X_train[:train_len], y_train[:train_len], X_train[train_len:], y_train[train_len:]
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test
