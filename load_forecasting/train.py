# ==============================================================================
# Desc : Script to perform model training and hyperparam selection
# ==============================================================================
import pandas as pd 
import numpy as np
from utils.utils import windowed_dataset, generate_cyclic_features, train_val_test_split, normalise, torch_dataset


if __name__ == "__main__":

    # ==========================================================================
    # Load Data, Feature Creation, Train-Val-Test split & Normalise Data
    # ==========================================================================

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

    for (X,y) in train_loader:
        # X : (64, 27), y : (64,1) where batch_size = 64
        pass
