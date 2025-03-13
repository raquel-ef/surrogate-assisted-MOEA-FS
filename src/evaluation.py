import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout, InputLayer # type: ignore


def calculate_errors(test_y, pred):
    """
    Calculate Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Correlation Coefficient (CC).
    
    Parameters:
    test_y (array-like): True target values.
    pred (array-like): Predicted values.
    
    Returns:
    tuple: (mae, rmse, cc) where:
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - cc: Correlation Coefficient
    """
    mae = mean_absolute_error(test_y, pred)
    rmse = np.sqrt(mean_squared_error(test_y, pred))
    cc = np.corrcoef(test_y.T, pred.T)[1,0]    
    return mae, rmse, cc


def normalized_means(result_steps, sizes, maximize=False):
    """
    Normalize error metrics from h-steps-ahead predictions to allow model comparison.
    
    Parameters:
    result_steps (list of arrays): Error metrics for h-steps-ahead predictions.
    sizes (list of int): List of sizes used for splitting the data.
    maximize (bool, optional): If True, inverts the metric (for correlation coefficient normalization).
    
    Returns:
    list: Normalized mean values for each split section.
    """
    ms = []
    minim = 0
    maxim = 1

    # Concatenate the error metrics into a single array, using list accumulation
    for data in result_steps:
        ms.append(data if not maximize else 1 - data)  # Avoid double concatenation
    ms = np.concatenate(ms)

    # Normalize the array
    norm = (ms - minim) / (maxim - minim)

    # Split and calculate means more efficiently
    norm = np.array_split(norm, np.cumsum(sizes)[:-1])  # Efficient splitting based on cumulative sizes
    meanList = [np.round(np.mean(i), 6) for i in norm]

    return meanList


def calculate_H(df, n_steps):
    """
    Calculate a summary metric "H" based on normalized error metrics (RMSE, MAE, CC).
    
    Parameters:
    df (DataFrame): Data containing RMSE, MAE, and CC values for different steps ahead.
    n_steps (int): Number of prediction steps.
    
    Returns:
    DataFrame: The original DataFrame with additional normalized metrics and H score.
    """

    # Filter out rows where 'RMSE StepsAhead' is empty
    dfSteps = df[df['RMSE StepsAhead'].str.len() > 0]

    # Extract and normalize the metrics using list comprehensions
    stepsRMSE = np.array([np.array(x[1:]) for x in dfSteps['RMSE StepsAhead']])
    stepsMAE = np.array([np.array(x[1:]) for x in dfSteps['MAE StepsAhead']])
    stepsCC = np.array([np.array(x[1:]) for x in dfSteps['CC StepsAhead']])

    sizes = [n_steps] * len(dfSteps)

    rmse = normalized_means(stepsRMSE, sizes)
    mae = normalized_means(stepsMAE, sizes)
    cc = normalized_means(stepsCC, sizes, maximize=True)

    # Combine the normalized results and calculate H
    normalizedMean = pd.DataFrame({
        'RMSEnorm': rmse,
        'MAEnorm': mae,
        'CCnorm': cc
    })
    normalizedMean['H'] = normalizedMean.mean(axis=1)

    # Set the index to match dfSteps
    normalizedMean.set_index(dfSteps.index, inplace=True)

    # Concatenate the original dfSteps with the normalizedMean DataFrame
    return pd.concat([dfSteps, normalizedMean], axis=1)


def predictions_h_stepsahead(testX, testy, model, n_steps):
    """
    Perform h-steps-ahead predictions using a machine learning model.
    
    Parameters:
    testX (DataFrame): Feature set for predictions.
    testy (DataFrame): True target values.
    model (Model): Trained machine learning model.
    n_steps (int): Number of steps ahead for prediction.
    
    Returns:
    tuple:
        - DataFrame: Error metrics (RMSE, MAE, CC) for each prediction step.
        - DataFrame: Predictions for each step ahead.
        - DataFrame: Updated testX with lagged predictions.
    """
    # Extract lags from column names
    predicted_attribute = testy.columns[0]
    lag_columns = testX.filter(regex=(f"{predicted_attribute}.*")).columns
    selected_lags = [int(col.split("_")[2]) for col in lag_columns]

    # Reset indices for test data
    test_X = testX.reset_index(drop=True)
    test_y = testy.reset_index(drop=True)

    # Initialize results DataFrame
    predictions = pd.DataFrame(index=test_X.index)
    results = []

    # 1-step ahead prediction
    predictions["pred1"] = model.predict(test_X).ravel()
    mae, rmse, cc = calculate_errors(test_y, predictions[["pred1"]])
    results.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

    # Handle case when no lagged variables exist
    if not selected_lags:
        print("No lagged variables found")


    # Multi-step ahead predictions (when lagged variables exist)
    for step in range(2, n_steps + 2):
        for lag in range(1, step):
            shift_lag = step - lag
            if shift_lag == 1 and 1 not in selected_lags:
                # Replace first lag with the smallest available lag
                col_name = f"Lag_{predicted_attribute}_{selected_lags[0]}"
                test_X[col_name] = predictions[f'pred{lag}'].shift(shift_lag)
            elif shift_lag in selected_lags:
                col_name = f"Lag_{predicted_attribute}_{shift_lag}"
                test_X[col_name] = predictions[f'pred{lag}'].shift(shift_lag)

        # Drop NaN values before prediction
        valid_X = test_X.dropna()
        pred = model.predict(valid_X.to_numpy()).ravel()

        # Insert NaN padding for alignment
        predictions[f'pred{step}'] = np.concatenate((np.full(step - 1, np.nan), pred))

        # Calculate errors
        valid_preds = predictions[f'pred{step}'][step - 1:]
        mae, rmse, cc = calculate_errors(test_y.iloc[step - 1:], valid_preds.to_frame())
        results.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, predictions, test_X


def predictions_h_stepsahead_LSTM(testX, testy, model, n_steps):
    """
    Perform h-steps-ahead predictions using an LSTM model.
    
    Parameters:
    testX (DataFrame): Feature set for predictions.
    testy (DataFrame): True target values.
    model (LSTM Model): Trained LSTM model.
    n_steps (int): Number of steps ahead for prediction.
    
    Returns:
    tuple:
        - DataFrame: Error metrics (RMSE, MAE, CC) for each prediction step.
        - DataFrame: Predictions for each step ahead.
        - DataFrame: Updated testX with lagged predictions.
    """

    test_X = testX.reset_index(drop=True)
    test_y = testy.reset_index(drop=True)
    predicted_attr = test_y.columns[0]
    
    # Extract lags from column names
    listaAtribSelected = sorted([int(col.split("_")[2]) for col in test_X.filter(regex=(predicted_attr + ".*")).columns])

    # Initialize results DataFrame
    predicciones = pd.DataFrame(index=test_X.index)
    dfResultados = []

    if listaAtribSelected:  # Ensure there are lagged target variables for step-ahead predictions
        # 1-step ahead prediction
        x_reshape = test_X.to_numpy().reshape(test_X.shape[0], 1, test_X.shape[1])
        predicciones["pred1"] = model.predict(x_reshape, verbose=0).ravel()

        rmse, mae, cc = calculate_errors(test_y, predicciones[['pred1']])
        dfResultados.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

        # h-step ahead predictions
        for i in range(2, n_steps + 2):
            for j in range(1, i):
                lag = i - j
                if lag == 1 and 1 not in listaAtribSelected:
                    first_lag = listaAtribSelected[0]
                    test_X[f"Lag_{predicted_attr}_{first_lag}"] = predicciones[f'pred{j}'].shift(lag)
                elif lag in listaAtribSelected:
                    test_X[f"Lag_{predicted_attr}_{lag}"] = predicciones[f'pred{j}'].shift(lag)

            arrayX = test_X.dropna().to_numpy().reshape(-1, 1, test_X.shape[1])
            predNa = np.insert(model.predict(arrayX, verbose=0), 0, [np.nan] * (i - 1))

            predicciones[f'pred{i}'] = predNa[:len(predNa)]

            rmse, mae, cc = calculate_errors(test_y.iloc[(i-1):], predicciones[[f'pred{i}']].iloc[(i-1):])
            dfResultados.append({'RMSE': rmse, 'MAE': mae, 'CC': cc})

    return pd.DataFrame(dfResultados), predicciones, test_X
    

def train_evaluate_lstm_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, n_neurons, batch_size, epochs, n_steps):
    """
    Trains an LSTM model on the given training data and evaluates it on 
    training, validation, and test sets using multi-step-ahead forecasting.

    Parameters:
    ----------
    train_X (DataFrame): Feature matrix for training.
    train_Y (DataFrame): Target variable for training.
    val_X (DataFrame): Feature matrix for validation.
    val_Y (DataFrame): Target variable for validation.
    test_X (DataFrame): Feature matrix for testing.
    test_Y (DataFrame): Target variable for testing.
    n_neurons (int): Number of neurons in the LSTM layer.
    batch_size (int): Number of samples per batch during training.
    epochs (int): Number of training epochs.
    n_steps (int): Number of steps ahead for prediction.

    Returns:
    -------
    DataFrame: A DataFrame containing RMSE, MAE, CC, and H scores for 
    train, validation, and test sets.

    Notes:
    ------
    - The function defines and trains an LSTM model with a single LSTM layer, 
      dropout for regularization, and a dense output layer.
    - It evaluates the model using RMSE, MAE, and Correlation Coefficient (CC) 
      for multi-step-ahead predictions.
    - The H metric is computed to assess prediction quality.
    """
    
    # Model Definition
    model = Sequential([
        InputLayer(shape=(1, train_X.shape[1])),
        LSTM(units=n_neurons, activation='relu', return_sequences=True),
        Dropout(0.2),
        Dense(1, activation="linear")
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', 
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Train the model
    model.fit(train_X.to_numpy().reshape(train_X.shape[0], 1, train_X.shape[1]), 
              train_Y, batch_size=batch_size, epochs=epochs, verbose=0)

    # Dictionary to store results
    datasets = {
        "Train": (train_X, train_Y),
        "Validation": (val_X, val_Y),
        "Test": (test_X, test_Y),
    }

    results = {}

    # Loop through datasets
    for key, (X, Y) in datasets.items():
        df_results, _, _ = predictions_h_stepsahead_LSTM(X, Y, model, n_steps)
        df_results = pd.DataFrame({
            'RMSE StepsAhead': [np.round(np.asanyarray(df_results['RMSE']), 6)],
            'MAE StepsAhead': [np.round(np.asanyarray(df_results['MAE']), 6)],
            'CC StepsAhead': [np.round(np.asanyarray(df_results['CC']), 6)]
        })
        df_results = calculate_H(df_results, n_steps)
        results[key] = df_results[['RMSE StepsAhead', 'MAE StepsAhead', 'CC StepsAhead', 'H']]

    # Combine results into one DataFrame
    df = pd.concat(results.values(), axis=1)
    df.columns = [
        'RMSE StepsAhead Train', 'MAE StepsAhead Train', 'CC StepsAhead Train', 'H Train',
        'RMSE StepsAhead Val', 'MAE StepsAhead Val', 'CC StepsAhead Val', 'H Val',
        'RMSE StepsAhead Test', 'MAE StepsAhead Test', 'CC StepsAhead Test', 'H Test'
    ]

    return df