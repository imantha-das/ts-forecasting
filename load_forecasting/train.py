# ==============================================================================
# Desc : Script to perform model training and hyperparam selection
# ==============================================================================
import pandas as pd 


if __name__ == "__main__":
    df = pd.read_csv("data/load_ammended.csv")
    print(df.head())