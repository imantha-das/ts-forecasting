# ==============================================================================
# Desc : Script to extract day of week and hour columns from existing data which 
# only contains the energy load value
# ==============================================================================
import pandas as pd 
from dateutil.relativedelta import relativedelta

def load_dataset(path:str)->pd.DataFrame:
    """
    Desc : Reads .xlsx data to pandas dataframe
    Inputs:
        - path to file
    Outputs
        - DataFrame 
    """
    df = pd.read_excel(path, header = None)
    df.rename(columns = {0 : "energy_load"}, inplace = True)
    return df 

def add_hour_dow(df:pd.DataFrame, last_date:str = "2022-05-15 23:00:00", endpoint:int=31)-> pd.DataFrame:
    """
    Desc : Adds a synthetic timestamp (these are not actual values) and extract days of week and hour columns
    Inputs:
        - DataFrame
        - last_date : add the specified date to the last point of time series and backtracks dates for previous time series points
        - endpoint : attempts to capture full seasonal cycles in data
    Outputs:
        - DataFrame
    """
    df = df.copy()
    df = df[:-endpoint]
    last_ts = pd.to_datetime(last_date, format = "%Y-%m-%d %H:%M:%S")
    ts = [last_ts - relativedelta(hours=  x) for x in range(df.shape[0])]
    ts.sort()
    df["date"] = ts
    df = df.assign(
        day_of_week = lambda x: x.date.dt.dayofweek,
        hour = lambda x: x.date.dt.hour
    )
    df.drop(columns = ["date"], inplace = True)
    df = df[["day_of_week", "hour", "energy_load"]]
    return df

if __name__ == "__main__":
    df = load_dataset(path = "data/load.xlsx")
    df = add_hour_dow(df)
    df.to_csv(path_or_buf="data/load_ammended.csv", index = False)

