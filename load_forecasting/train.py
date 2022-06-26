# ==============================================================================
# Desc : Script to perform model training and hyperparam selection
# ==============================================================================
import pandas as pd 
import numpy as np
from utils.utils import windowed_dataset, generate_cyclic_features, train_val_test_split, normalise, torch_dataset
from models import LSTM

# Check if you can remove this
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Hyperparams
    # --------------------------------------------------------------------------
    window_size = 24
    batch_size = 64
    val_ratio = 0.15
    test_ratio = 0.15

    epochs = 10
    model_name = "lstm"
    hidden_size = 64
    num_layers = 1
    optimizer_name = "adam"
    lr = 0.001

    # --------------------------------------------------------------------------
    # Load Dataset -> Feature Exctraction -> Train-Val-Test split -> Normalise -> Tensor Dataset
    # --------------------------------------------------------------------------

    # import data
    df = pd.read_csv("data/load_ammended.csv")

    # Generate sin_hour, cos_hour
    # Replace hour with these components as NN will inherently learn better.
    df = generate_cyclic_features(df, "hour", 24)

    # Windowed dataset - removes last incomplete window
    X,y = windowed_dataset(seq = df["energy_load"], ws= window_size)

    # Add time features to windowed dataset - :len(X) -> removes the respective hour DOW incomplete window 
    dow_hr = df[["day_of_week", "sin_hour", "cos_hour"]][:len(X)].values

    # Stack features
    X = np.hstack((X, dow_hr))

    # Train - Validation - Test Split 
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, val_ratio, test_ratio)

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
        batch_size = batch_size
    )

    # --------------------------------------------------------------------------
    # Model Training
    # --------------------------------------------------------------------------
    models = {"lstm" : LSTM}
    model = models[model_name](input_size = 27, hidden_size = hidden_size, num_layers = num_layers)
    optimizers = {"adam" : optim.Adam(model.parameters(), lr = lr)}
    optimizer = optimizers[optimizer_name]
    criterion = nn.MSELoss()

    training_losses = []
    val_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for (X_batch,y_batch) in train_loader:
            # SHAPES : X - (64, 27), y - (64,1) where batch_size = 64
            # LSTM model requires, (N, L, H_in)
            X_batch = X_batch.unsqueeze(1) # Add dimension at 1, for L
            yhat,_ = model.forward(X_batch) # (64,1) ... for each iteration

            # Compute Loss
            loss = criterion(yhat, y_batch)
            # Keep track all losses in batch
            batch_losses.append(loss.item())

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Keep track of mean loss in batch
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)
        
        # Validation losses
        with torch.no_grad():
            val_batch_losses = []
            for (X_batch_val, y_batch_val) in val_loader:
                X_batch_val = X_batch_val.unsqueeze(1)
                yhat_val, _ = model.forward(X_batch_val)

                val_batch_losses.append(criterion(yhat_val, y_batch_val).item())
            val_loss = np.mean(val_batch_losses)
            val_losses.append(val_loss)

        if (epoch <= 10) | (epoch%50 == 0):
                print(f"[{epoch}/{epochs}] Training loss : {training_loss:.4f}\t Validation loss : {val_loss:.4f}")
                