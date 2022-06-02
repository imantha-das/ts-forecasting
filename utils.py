import os
import numpy as np 
import pandas as pd

from dateutil.relativedelta import relativedelta

import tensorflow as tf

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

def plot_series(x, y, fig, yaxis_title):
    """
    Plots time series
    Inputs 
        - x : dates (np.array{datetime})
        - y : sequence (np.array{float})
        - fig : Plotly Figure object (plotly.Figure)
        - yaxis_title : title of yaxis (str)
    Outputs
        - Plotly figure (Plotly.Figure)
    """
    fig.add_trace(go.Scatter(
        x = x,
        y = y,
        name = "original_ts"
    ))
    fig.update_layout(xaxis_title = "date", yaxis_title = yaxis_title, width = 1500, height = 800, template = 'plotly_white')
    return fig


def split_dataset(seq,split_time):
    """ 
    Splits data to training and testing sets
    Inputs
        - seq : sequence (np.array{float})
        - split_time : At which point to split data for training (Int)
    Outputs
        - (train_set,test_set) (Tuple{np.array,np.array})
    """
    train_set = seq[:split_time]
    test_set = seq[split_time:]
    return (train_set, test_set)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer, shuffle = True):
    """
    Generates dataset windows
    Inputs:
        - series (array of floats) : contains the values of the timeseries
        - window_size (int) : the number of time steps to include in the feature
        - batch_size (int) : the batch size
        - shuffle_buffer (TD Dataset) : TF Dataset containing time windows
    Outputs:
        - dataset : Training data in the form of a sliding window (Tensorflow Dataset Object)
    """

    # Generates a Tf Dataset from the series
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift = 1, drop_remainder = True)

    # Flatten the window by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    if shuffle:
        # Shuffle the windows
        dataset = dataset.shuffle(shuffle_buffer)

        # Create batches of windows
        dataset = dataset.shuffle(shuffle_buffer)

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

def tune_learning_rate(model,trainset, lr = 1e-8):
    """
    Tunes learning rate
    Inputs
        - model : Tensorflow NN Model (Tensorflow Model)
        - trainset : Train data (Tensorflow Dataset Object)
        - lr: learning rate (float)
    Outputs
        - history : loss function 
    """

    # Set the learning rate scheduler
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: lr * 10**(epoch/20)
    )

    # Initialise the optimizer
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)

    # Set the training parameters
    model.compile(loss = tf.keras.losses.Huber(), optimizer = optimizer)

    # Train the model
    history = model.fit(trainset, epochs = 100, callbacks = [lr_schedule])

    # Learning-rate plot
    lrs = lr * (10 ** (np.arange(100)/20))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = lrs,
        y = history.history["loss"]
    ))
    fig.update_xaxes(type = "log")
    fig.update_layout(xaxis_title = "learning-rate", yaxis_title = "loss", width = 1500, height = 800, template = 'plotly_white')

    best_lr = lrs[np.argmin(history.history["loss"])]

    return history,fig, best_lr 
