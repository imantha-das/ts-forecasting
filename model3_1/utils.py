import os
import numpy as np 
import pandas as pd

import holidays
from dateutil.relativedelta import relativedelta

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objs as go

# ==============================================================================
# Loading Data
#todo -- xlsx should give weekday or not
# ==============================================================================

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

# ==============================================================================
# Generating time lags
# ==============================================================================

def generate_time_lags(df, n_lags):
    """Desc : """
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        df_n[f"lag{n}"] = df_n["value"].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n

# ==============================================================================
# Generating datetime features
#todo  - Need to change the method of reading these features
# ==============================================================================

def generate_time_features(df):
    df_features = df.copy()
    df_features = (
    df_features
    .assign(hour = df.index.hour)
    .assign(day_of_week = df.index.dayofweek)
    )
    return df_features

# ==============================================================================
# Generate Cyclic features
#todo --make sure hour of day is a given feature
# ==============================================================================

def generate_cyclic_features(df, col_name, period, start_num = 0):
    kwargs = {
        f"sin_{col_name}" : lambda x: np.sin(2 * np.pi*(df[col_name] - start_num)/period),
        f"cos_{col_name}" : lambda x: np.cos(2 * np.pi*(df[col_name] - start_num)/period)
    }
    return df.assign(**kwargs).drop(columns = [col_name])

# ==============================================================================
# Holidays
#* NOT used in current model
#todo --need to sort out this two functions in a better manner
# ==============================================================================


def is_holiday(date):
    us_holidays = holidays.Singapore()
    date = date.replace(hour = 0)
    return 1 if (date in us_holidays) else 0

def add_holiday_col(df, holidays):
    return df.assign(is_holiday = df.index.to_series().apply(is_holiday))


# ==============================================================================
# Train-Validation-Test sets
# ==============================================================================
def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns = [target_col])
    return X, y 

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle = False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle = False)
    return X_train, X_val, X_test, y_train, y_val, y_test 

# ==============================================================================
# Normalise
# ==============================================================================

def normalise(X_train, X_val, X_test, y_train, y_val, y_test):
    normaliser = MinMaxScaler()
    X_train_norm = normaliser.fit_transform(X_train)
    X_val_norm = normaliser.fit_transform(X_val)
    X_test_norm = normaliser.fit_transform(X_test)

    y_train_norm = normaliser.fit_transform(y_train)
    y_val_norm = normaliser.fit_transform(y_val)
    y_test_norm = normaliser.fit_transform(y_test)

    return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, normaliser