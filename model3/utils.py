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
    

