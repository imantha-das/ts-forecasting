# ==============================================================================
# Utility functions for train.py
# ==============================================================================
import pandas as pd 
import numpy as np

# ------------------------------------------------------------------------------
# Time lag based features 
# ------------------------------------------------------------------------------

def get_time_lags(df:pd.DataFrame,lags:int)->np.array:
    """
    Desc : Creates a sliding window based on the specified lags
    Inputs :
        df : DataFrame
        lags : last n values of lags as specified (sliding window)
    Outputs :
        X : an array of windows
    """
    pass

