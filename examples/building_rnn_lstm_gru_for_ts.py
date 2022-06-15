import os 
import pandas as pd 
import numpy as np 

import plotly.graph_objs as go 
import plotly.io as pio 

pio.templates.default = "plotly_white"

from datetime import datetime

import torch 
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

import holidays

from termcolor import colored

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Read Data
def read_data(path):
    
    df = pd.read_csv(path_to_data)
    df = df.sort_values(by = "Datetime", axis = 0) # Data messed up require sorting
    df.set_index(["Datetime"], inplace = True)
    df.index = pd.to_datetime(df.index, format = "%Y-%m-%d %H:%M:%S")
    df.rename(columns = {"AEP_MW" : "value"}, inplace = True)
    return df
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot time series
def plot_original_ts(df, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df.index,
        y = df[y],
        mode = "lines",
        line = dict(color = "dodgerblue"),
    ))
    fig.update_layout(xaxis_title = "date", yaxis_title = "AEP_hourly")
    return fig

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Generate time lags
def generate_time_lags(df, n_lags, n_lag_col):
    """
    Generate time lags
    Inputs:
        - df : DataFrame
        - n_lags : number of lags
        - n_lag_col : column to perform lags on
    Outputs:
        - df_n : DataFrame with time lags as seperate columns, orginal columns not dropped
    """
    df_n = df.copy()
    for n in range(1, n_lags + 1):
        # Up until n_lags the dataframe will be 
        df_n[f"lag{n}"] = df_n[n_lag_col].shift(n)
        # select columns from n_lags onwards, in otherwords if lags are 200
    df_n = df_n.iloc[n_lags:]
    return df_n

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_timestamp_features(df):
    """
    Generates timestamp based features : hour, day, month, dayofweek, week
    Inputs :
        - df : DataFrame with DateTime as index
    Outputs :
        - df : DataFrame, with Datetime Features
    """
    df_features = df.assign(
        hour = df.index.hour,
        day = df.index.day,
        month = df.index.month,
        day_of_week = df.index.dayofweek,
        week_of_year = df.index.isocalendar().week
    )
    return df_features

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_cyclic_features(df, col_name, period, start_num):
    """
    Generates cyclic features on columns such as hour
    Inputs :
        - df : DataFrame
        - col_name : column you want cyclic features to appled to
        - period : period of the feature, i.e hour = 24
        - start_num : starting number, i.e 0
    """
    kwargs = {
        f"sin_{col_name}" : lambda x: np.sin(2 * np.pi*(df[col_name] - start_num)/period),
        f"cos_{col_name}" : lambda x: np.cos(2 * np.pi*(df[col_name] - start_num)/period)
    }
    return df.assign(**kwargs).drop(columns = [col_name])

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def is_holiday(date, holiday_country):
    date = date.replace(hour = 0)
    return 1 if (date in holiday_country) else 0

def add_holiday_col(df, holiday_country):
    return df.assign(
        is_holiday = df.index.to_series().apply(lambda x: is_holiday(x, holiday_country))
        )

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# One-Hot-Encoding
def one_hot_encode(df, col_names):
    """
    One hot encode assigned columns
    Inputs:
        - df : DataFrame
        - col_names = (list) of columns
    Output:
        - df : DataFrame 
    """
    df = df.copy()
    column_trans = make_column_transformer(
        (OneHotEncoder(),col_names),
        remainder= "passthrough"
    )
    one_hot_arr = column_trans.fit_transform(df)
    column_names = [name.split("__")[1] for name in column_trans.get_feature_names_out()]

    print(colored(one_hot_arr.shape, "magenta"))
    

    df_one_hot = pd.DataFrame(data = one_hot_arr, columns=column_names)
    print(colored(df_one_hot.columns.values, "magenta"))
    return df_one_hot


# =============================================================================================
if __name__ == "__main__":
    #print(colored(f"cwd : {colored(os.getcwd(), 'magenta')}"))
    path_to_data = os.path.join("examples","data","AEP_hourly.csv")
    # Get Data
    df = read_data(path = path_to_data)
    #print(df)

    # Plot timeseries 
    fig = plot_original_ts(df, "value")
    #fig.show()

    # Generate timestamp feature
    df_features = generate_timestamp_features(df)
    
    # Generate Cyclic features
    df_features = generate_cyclic_features(df_features, "hour", 24, 0)

    # Add Holidays
    us_holidays = holidays.US()
    df_features = add_holiday_col(df_features, us_holidays)

    #print(df_features.columns.values)
    """
    df_test = pd.DataFrame({
        "x" : np.random.rand(10),
        "y" : np.random.rand(10),
        "z1" : np.random.randint(1,3,10),
        "z2" : np.random.randint(1,4,10)
    })

    df_test = one_hot_encode(df_test, ["z1", "z2"])
    print(df_test)
    """
    
    df_features = one_hot_encode(df_features, ["month", "day_of_week", "week_of_year"])

    print(df_features)

    #df_features = one_hot_encode(df_features, col_names=["month", "day", "day_of_week", "week_of_year"])
    #print(df_features.head())
