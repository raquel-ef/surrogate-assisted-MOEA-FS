import numpy as np
import pandas as pd
from sklearn import preprocessing
from skrebate import ReliefF


# Applies sliding window transformation method keeping the attributes original order and deleteing the original attributes
def lags(df_original, n_lag):
    """
    Applies sliding window transformation keeping the attributes original order 
    and removing the original attributes.
    
    Parameters:
    df_original (DataFrame): Feature set for predictions.
    n_lag (int): Number of lags.
    
    Returns:
    DataFrame: New dataframe with the sliding window transformation.
    """
    df = df_original.copy()
    lagged_data = []

    # Generate lagged columns while maintaining original order
    for col in df.columns: 
        base_name = col.split(' ', 1)[0]  # Extract base name
        for j in range(1, n_lag + 1):
            lagged_data.append(df[col].shift(j).rename(f'Lag_{base_name}_{j}'))

    # Concatenate lagged columns
    df_lags = pd.concat(lagged_data, axis=1)

    # Append the last column from the original dataset
    df_lags[df.columns[-1]] = df[df.columns[-1]]

    return df_lags


def get_correlation_rank(train_X, train_Y, n_attributes):
    """
    Computes the correlation-based ranking of features using standardized data.

    Parameters:
    train_X (DataFrame): Feature dataset.
    train_Y (DataFrame): Target dataset.
    n_attributes (int): Number of attributes to consider.

    Returns:
    np.ndarray: Normalized correlation scores for features.
    """
    # Standardize data to apply Cfs method
    trainNorm = train_X.copy()
    trainNorm[train_Y.columns[0]] = train_Y

    NO2WS7stand=pd.DataFrame(preprocessing.StandardScaler().fit_transform(trainNorm))

    cor = NO2WS7stand.corr()
    corOutput = cor[n_attributes][:-1].values

    return (corOutput-min(corOutput))/(max(corOutput)-min(corOutput))


def get_relieff_rank(train_X, train_Y, n_attributes, n_neighbors, n_jobs):
    """
    Computes feature importance scores using the ReliefF algorithm.

    Parameters:
    train_X (DataFrame): Feature dataset.
    train_Y (DataFrame): Target dataset.
    n_attributes (int): Number of attributes to select.
    n_neighbors (int): Number of neighbors to consider in ReliefF.
    n_jobs (int): Number of parallel jobs to run.

    Returns:
    tuple: Raw ReliefF scores and their normalized values.
    """
    reliefF = ReliefF(n_features_to_select=n_attributes, n_neighbors=n_neighbors, n_jobs=n_jobs)
    reliefF.fit(np.array(train_X), np.array(train_Y).ravel())

    # Sort importances from best to worse
    reliefFscores = reliefF.feature_importances_

    # Normalize values
    return reliefFscores, (reliefFscores-min(reliefFscores))/(max(reliefFscores)-min(reliefFscores))


def get_dataset_best_H(dataset, dfSolutions):
    """
    Retrieves the best feature subset based on the lowest "H" score.

    Parameters:
    dataset (DataFrame): Original dataset.
    dfSolutions (DataFrame): Dataframe containing different feature selection solutions.

    Returns:
    DataFrame: Dataset with the best selected attributes.
    """
    # Identify the row with the best "H" score
    best_solution = dfSolutions.loc[dfSolutions['H'].idxmin()]

    # Extract selected attributes
    selected_attributes = best_solution['SelectedAttrib']

    # If 'SelectedAttrib' is stored as a string, convert it into a list
    if isinstance(selected_attributes, str):
        selected_attributes = eval(selected_attributes)  # Convert string representation of list to an actual list

    if isinstance(selected_attributes, np.ndarray):
        selected_attributes = selected_attributes.tolist()

    selected_attributes.append(dataset.columns[-1])

    # Create a new dataset using the selected attributes
    return dataset[selected_attributes]
