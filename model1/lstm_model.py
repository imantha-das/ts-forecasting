import os
import numpy as np
import pandas as pd 
from utils import plot_series, split_dataset, windowed_dataset, tune_learning_rate, load_data
import plotly.graph_objs as go

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Lambda

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fit Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def fit_model(model,trainset, lr, init_weights, epochs):
    """
    Function to fit model
    Inputs
        - Tensorflow model
        - trainset : array, dims = 1 (np.array)
        - lr : learning rate (float)
        - init_wights : Initial Weights (float)
        - epochs : number of epochs for training (int)
    Outputs
        - history : model history (dict)
        - fig : figure show loss and mae (plotly figure)
    """
    # Reset states genertated by keras and reset weights
    tf.keras.backend.clear_session()
    model.set_weights(init_weights)

    # Initialize optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.9)

    # Set the training parameters
    model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer, metrics = ["mae"])

    # Train the model
    history = model.fit(trainset, epochs = epochs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = np.arange(0, epochs),
        y = history.history["loss"],
        name = "loss"
    ))
    fig.add_trace(go.Scatter(
        x = np.arange(0, epochs),
        y = history.history["mae"],
        name = "mae"
    ))

    fig.update_layout(xaxis_title = "epochs", yaxis_title = "loss/mae", width = 1500, height = 800, template = 'plotly_white')

    return history, fig

# Forecast 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def model_forecast(model, series, window_size, batch_size):
    """
    Uses an input model to generate predictions on data windows

    Args :
        - model (TF Keras Model) - model that accepts data windows
        - series (array of floats) - contains the values of the time series
        - window_size (int) - the number of time steps to include in the window
        - batch_size (int) - the batch-size

    Returns:
        forecast (np.array) - array containing predictions
    """
    #Generate a TF Dataset from series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift = 1, drop_remainder = True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    # Create batches of the window
    dataset = dataset.batch(batch_size).prefetch(1)

    # Get the prediction on the entire dataset
    forecast = model.predict(dataset)
    return forecast

# Main 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    # Load Dataset
    # --------------------------------------------------------------------------------------------------
    # Notes
    #   - you will have to remove `index_col = 0` for other datasets
    file_path = "ts-forecasting/data/load.xlsx"
    df = load_data(path = file_path)

    # Plot original TS
    p1 = go.Figure()
    p1 = plot_series(df["date"], df["y"], p1, "energy load")
    #p1.show()

    # Test - Train Split
    # -------------------------------------------------------------------------------------------------
    # Notes
    #   -split_dataset(seq = np.array , ...)

    split_time = 3000

    X_train, X_valid = split_dataset(seq = df.y.values, split_time = split_time)
    (window_size, batch_size, shuffle_buffer_size) = (30,32,1000)

    # Sliding window
    trainset = windowed_dataset(
        series = X_train, 
        window_size = window_size, 
        batch_size = batch_size, 
        shuffle_buffer = shuffle_buffer_size, 
        shuffle = False
    )

    # LSTM Model - Tensorflow
    # --------------------------------------------------------------------------------------------------
    model =  Sequential([
        Conv1D(filters = 64, kernel_size = 3, strides = 1, activation = "relu", padding = "causal", input_shape = [window_size, 1]),
        LSTM(64, return_sequences = True),
        LSTM(64),
        Dense(30, activation = "relu"),
        Dense(30, activation = "relu"),
        Dense(1),
        Lambda(lambda x: x * 400)
    ])

    # --- Tune learning rate ---
    # Store initial weights
    init_weights = model.get_weights()

    # Find best learning rates for these weights
    history, p2, best_lr = tune_learning_rate(model = model, trainset = trainset, lr = 1e-8)
    p2.show()

    print(f"Best learning-rate : {best_lr}")
    #best lr = 5.011872336272723e-07

    # Train Model
    # -------------------------------------------------------------------------------------------------
    # Reset states and weights and generated by keras

    history, p3 = fit_model(model = model, trainset = trainset, lr = best_lr, init_weights = init_weights, epochs = 100)
    p3.show()

    # Prediction 
    # ------------------------------------------------------------------------------------------------
    # Take last window (from training set) + rest of data (validation set) 
    forecasted_series = df.y.values[split_time - window_size : -1]
    # prediction
    forecast = model_forecast(model = model, series = forecasted_series, window_size = window_size, batch_size = batch_size)
    # drop single dimension axis
    results = forecast.squeeze() 

    p1.add_trace(go.Scatter(
        x = df.date.values[split_time : -1],
        y = results,
        mode = "lines",
        name = "forecasted Ts"
    ))

    p1.show()

    print(f"MSE : {mean_squared_error(y_true=X_valid, y_pred = results)}")
    print(f"MAE : {mean_absolute_error(y_true=X_valid, y_pred = results)}")

