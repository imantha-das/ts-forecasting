# ==============================================================================
# Imports
# ==============================================================================
print("Hello")

import os 
import copy 
import sys 
from datetime import datetime

import numpy as np 
import pandas as pd 

from termcolor import colored
from torch import norm, normal

import nni
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.graph_objs as go

#os.chdir("cryo-polygen/ts-forecasting")

# ==============================================================================
# Set Current Path to ts-forecasting folder 
# ==============================================================================
# if os.path.basename(os.getcwd()) != "ts-forecasting":
#     print(f"cwd : {os.getcwd()}")
#     path = input("Enter path from cwd to ts-forecasting : \n")
#     os.chdir(os.path.join(os.getcwd(),path))
#     current_path = os.getcwd()

#     sys.path.append(current_path)
# else:
#     current_path = os.getcwd()
#     sys.path.append(current_path)

# ==============================================================================
# Rest of the imports
# ==============================================================================

from utils import load_data, generate_time_lags, generate_time_features, generate_cyclic_features
from utils import train_val_test_split, normalise
from lstm import LSTMModel
from gru import GRUModel

# ==============================================================================
# nni parameters
#* nni - get params from search.py
# ==============================================================================
search_params = nni.get_next_parameter()
learning_rate = search_params["learning_rate"]
hidden_size = search_params["hidden_size"]
optimizer_name = search_params["optimizer"]

# ==============================================================================
# Load Data
# ==============================================================================

#file_path = os.path.join(current_path, "data", "load.xlsx")
file_path = os.path.join("data", "load.xlsx")
df = load_data(file_path, last_date = "2022-05-15 23:00:00", endpoint=31)
df.rename(columns = {"y" : "value", "date" : "Datetime"}, inplace = True)
df.set_index("Datetime", inplace = True)
df = df[:3475] #Removed the messed up timeseries data

# ==============================================================================
# Feature Extraction
# ==============================================================================

df_generated = generate_time_lags(df, 24) # Adds time lags
df_features = generate_time_features(df_generated) #Adds hour and day of week
df_features = pd.get_dummies(df_features, columns = ["day_of_week"]) #one hot encode dayofweek columns
df_features = generate_cyclic_features(df_features, "hour", 24, 0)

# Merge the two dataframes
df_generated.drop(columns = "value", inplace = True)
df_features = pd.merge(df_features,df_generated, left_index = True, right_index = True)

# ==============================================================================
# Train-Validation-Test sets
# -dataframes
# ==============================================================================
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_features, "value", 0.1)

# print("shapes")
# print(f"X_train : {X_train.shape}")
# print(f"y_train : {y_train.shape}")
# print(f"X_val : {X_val.shape}")
# print(f"y_val : {y_val.shape}")
# print(f"X_test : {X_test.shape}")
# print(f"y_test : {y_test.shape}")

# ==============================================================================
# Normalise data
#   -np.arrays
# ==============================================================================
X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, normalizer = normalise(X_train, X_val, X_test, y_train, y_val, y_test)

# ==============================================================================
# Dataloader
# ==============================================================================

batch_size = 64

train_features = torch.Tensor(X_train_norm)
train_targets = torch.Tensor(y_train_norm)
val_features = torch.Tensor(X_val_norm)
val_targets = torch.Tensor(y_val_norm)
test_features = torch.Tensor(X_test_norm)
test_targets = torch.Tensor(y_test_norm)

train_data = TensorDataset(train_features, train_targets)
val_data = TensorDataset(val_features, val_targets)
test_data = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False, drop_last = True)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, drop_last = True)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, drop_last = True)

# ==============================================================================
# Define Model
#todo Need to get the model in a different way
# ==============================================================================

def get_model(model,  model_params):
    models = {
        "gru" : GRUModel,
        "lstm" : LSTMModel
    }
    return models.get(model.lower())(**model_params)

# ==============================================================================
# Optimizer
#todo will likely have figure out a way to call this function in a seperate way
# ==============================================================================

# Training
class Optimizer():
    def __init__(self, model, criterion, optimizer):
        self.model = model 
        self.criterion = criterion 
        self.optimizer = optimizer
        self.training_losses = []
        self.val_losses = []

    def train_step(self, X, y):
        # set the model to train mode
        self.model.train() # allows the weights of the network to be updated
        # Make predictions
        yhat = self.model.forward(X)
        # Compute loss
        loss = self.criterion(y, yhat)
        # Compute gradients
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Return losses
        return loss.item()

    def train(self, train_loader, val_loader, batch_size = 64, n_epochs = 50, n_features = 1):
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.view([batch_size,  -1, n_features]) # add a dimension at 2, (64, 106) --> (64,106,1)
                loss = self.train_step(X_batch,  y_batch) # Calls the training step herem which calls model.forward()
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.training_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]) # adds a dimension ar 2, (64, 106) --> 
                    self.model.eval()
                    yhat = self.model.forward(x_val) # Calls model.forward() method here
                    val_loss = self.criterion(y_val, yhat)
                    batch_val_losses.append(val_loss.item())
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch%50 == 0):
                print(f"[{epoch}/{n_epochs}] Training loss : {training_loss:.4f}\t Validation loss : {validation_loss:.4f}")

        torch.save(self.model.state_dict(),f"state_dict/example_model_{datetime.now()}")

    def evaluate(self, test_loader, batch_size = 1, n_features = 1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]) # (1, 106) --> (1, 106, 6)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().numpy())
                values.append(y_test.detach().numpy())

        return predictions, values

    def evaluate_train(self, train_loader, batch_size = 64, n_features = 1):
        with torch.no_grad():
            predictions = []
            values = []
            for x_train, y_train in train_loader:
                x_train = x_train.view([batch_size, -1, n_features])
                self.model.eval()
                yhat = self.model.forward(x_train)
                predictions.append(yhat.detach().numpy())
                values.append(y_train.detach().numpy())

        return predictions, values

# ==============================================================================
# Inverse Transform
# ==============================================================================

def inverse_transform(scaler,df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df

# ==============================================================================
# Prediction
# ==============================================================================

def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis = 0).ravel()
    preds = np.concatenate(predictions, axis = 0).ravel()
    df_result = pd.DataFrame(data = {"value" : vals, "predictions" : preds}, index = df_test.head(len(vals)).index)
    df_result = inverse_transform(scaler, df_result, [["value", "predictions"]])
    return df_result

# ==============================================================================
# Metrics
# ==============================================================================
def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df.value, df.predictions),
            'rmse' : mean_squared_error(df.value, df.predictions) ** 0.5,
            'r2' : r2_score(df.value, df.predictions)}


# ==============================================================================
# Training
# ==============================================================================
input_size = len(X_train.columns) # 106
output_size = 1
#hidden_size = 64 #* uncomment this if not running through search.py (nni)
layer_size = 2
batch_size = 64
dropout_prob = 0.2 
n_epochs = 500
#learning_rate = 1e-3 #* uncomment this if not running through search.py (nni)
weight_decay = 1e-6
model_params = {
    "input_size" : input_size,
    "hidden_size" : hidden_size,
    "layer_size" : layer_size, 
    "output_size" : output_size,
    "dropout_prob" : dropout_prob
}
optimizers = {
    "adam" : optim.Adam,
    "sgd" : optim.SGD,
    "adamax" : optim.Adamax
}

model = get_model("lstm", model_params)
criterion = nn.MSELoss(reduction = "mean")
#optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay) #* uncomment this if not running through search.py
optimizer = optimizers[optimizer_name](params = model.parameters(), lr = learning_rate, weight_decay = weight_decay)
opt = Optimizer(model = model, criterion = criterion, optimizer = optimizer)
opt.train(train_loader=train_loader, val_loader=val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_size)

# ==============================================================================
# Plots : Losses
# ==============================================================================

training_losses = opt.training_losses
validation_losses = opt.val_losses

p1 = go.Figure()
p1.add_trace(go.Scatter(y = training_losses, name = "training losses"))
p1.add_trace(go.Scatter(y = validation_losses, name = "validation losses"))
p1.update_layout(xaxis_title = "epochs", yaxis_title = "losses")
p1.update_layout(title = "LSTM - Mean Squred error on normalised data")
p1.show()


# ==============================================================================
# Prediction on testset
# ==============================================================================
predictions, values = opt.evaluate(test_loader, batch_size = 64, n_features = input_size)
df_result = format_predictions(predictions, values, X_test, normalizer) 
result_metrics = calculate_metrics(df_result)

print(result_metrics)

test_mse = result_metrics["rmse"]
nni.report_final_result(test_mse) #* Report performance on mse to nni

# ==============================================================================
# Plot : Prediction
# ==============================================================================
p2 = go.Figure()
p2.add_trace(go.Scatter(x = df_result.index, y = df_result.value, name = "original ts"))
p2.add_trace(go.Scatter(x = df_result.index, y = df_result.predictions, name = "lstm prediction"))
p2.update_layout(title = "Energy Load")
p2.show()