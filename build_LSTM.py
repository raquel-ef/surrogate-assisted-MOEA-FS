#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, InputLayer

import os
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import time
from src.data_processing import read_arff, split_features_target, to_numpy, reshape_timeseries
import config.config as config


# Set random seed for reproducibility
random.seed(config.SEED_VALUE)
np.random.seed(config.SEED_VALUE)
tf.random.set_seed(config.SEED_VALUE)


# Load dataset
DATA_PATH = os.path.join('.', 'data', config.DATASET_NAME)
dataset = read_arff(DATA_PATH)

# Split data into training (60%), validation (20%), and test (20%) sets
n = len(dataset)
train_set = dataset.iloc[: int(0.6 * n)]
val_set = dataset.iloc[int(0.6 * n) : int(0.8 * n)]
test_set = dataset.iloc[int(0.8 * n) :]

# Extract features and targets
train_X, train_Y = split_features_target(train_set)
val_X, val_Y = split_features_target(val_set)
test_X, test_Y = split_features_target(test_set)

# Normalize data using MinMaxScaler
scaler_X, scaler_Y = MinMaxScaler(), MinMaxScaler()
train_X = scaler_X.fit_transform(train_X)
train_Y = scaler_Y.fit_transform(train_Y)

val_X = scaler_X.transform(val_X)
val_Y = scaler_Y.transform(val_Y)

test_X = scaler_X.transform(test_X)
test_Y = scaler_Y.transform(test_Y)

# Convert to numpy arrays
train_X, train_Y = to_numpy(train_X), to_numpy(train_Y)
val_X, val_Y = to_numpy(val_X), to_numpy(val_Y)
test_X, test_Y = to_numpy(test_X), to_numpy(test_Y)

# Reshape data for LSTM
train_X_timeseries = reshape_timeseries(train_X)
val_X_timeseries = reshape_timeseries(val_X)
test_X_timeseries = reshape_timeseries(test_X)

start_time = time.time()

# Define LSTM model
model = Sequential([
    InputLayer(shape=train_X_timeseries.shape[1:]),
    LSTM(units=config.N_NEURONS, activation='relu', return_sequences=True),
    Dropout(0.2),
    Dense(1, activation="linear")
])

# Compile model using TensorFlow's built-in RMSE metric
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train model
model.fit(train_X_timeseries, train_Y, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=0, 
          validation_data=(val_X_timeseries, val_Y))

print(f"--- {(time.time() - start_time):.2f} seconds ---")

# Make predictions on test data
predictions = model.predict(test_X_timeseries, verbose=0).flatten()

# Evaluate model performance
mae = mean_absolute_error(test_Y.flatten(), predictions)
rmse = root_mean_squared_error(test_Y.flatten(), predictions)
r2 = r2_score(test_Y.flatten(), predictions)

# Print results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save trained model
os.makedirs('./models', exist_ok=True)
model.save(os.path.join('.', 'models', config.MODEL_NAME))
