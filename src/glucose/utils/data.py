"""src/glucose/utils/data.py

This scripts contains all functionality to process
and transform data into suitable formats for RNN and
LSTM networks.

Methods
-------
    scale_gcm_data
        scales the GCM data according to pre specified min_v and max_v values
    split_gcm_data
        splits the GCM data according to the RNN/LSTM format

"""

# Math imports
from itertools import islice

# Data science imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# Typing imports
from typing import Tuple, List


def split_gcm_datav2(cgm_list: np.array,
                     time_window: int,
                     pred_horizons: List[int],
                     pred_index: int = 0) -> Tuple[np.array, np.array]:
    """
    Splits the GCM data into a format which is applicable
    to use in training of LSTM and RNN networks.

    Parameters
    ----------
    cgm_list: np.array
        the CGM values (preferably transformed) in an np.array
    time_window: int
        the amount of steps to look back to make a prediction
    pred_horizons: List[int]
        the amount of steps to predict ahead
    pred_index: int
        in case of an array with multiple values, which index to use for the predictions?
        (e.g. we have time and glucose values we want to predict glucose values not time)

    Returns
    -------
    np.array, np.array
        returns two `np.array` containing the X and Y values for the specific sequence
    """

    x = []
    y = []

    for n in range(len(cgm_list) - time_window):
        if max(pred_horizons) + time_window + n < len(cgm_list):
            x.append(cgm_list[n:n + time_window])

            # Check if array is multi-dimensional
            if cgm_list.shape[1] > 1:
                # if so, we use the FIRST value as our Y-value to predict
                y.append([cgm_list[n + time_window + o - 1][pred_index]
                         for o in pred_horizons])
            else:
                # else just use the Series
                y.append([cgm_list[n + time_window + o - 1]
                         for o in pred_horizons])

    return np.array(x), np.array(y)


def scale_gcm_activpal_data(gcm_values: np.array,
                            scaler: Tuple[MinMaxScaler, MinMaxScaler] = None,
                            min_v: int = 0,
                            max_v: int = 1) -> Tuple[MinMaxScaler, MinMaxScaler, np.array]:
    """
    Scales the GCM data according to `min_v` and `max_v` values.
    Furthermore, this generates an extra y-value scaler to use with multivariate time series
    data.

    Parameters
    ----------
    gcm_values: np.array
        the raw GCM values
    scaler: MinMaxScaler
        if scaler is specified (e.g. in test values) use this object
    min_v: int, optional, standard = 0
        the minimal value of the transformation, standard 0
    max_v: int, optional, standard = 1
        the maximal value of the transformation, standard 1


    Returns
    -------
    Tuple[MinMaxScaler, pd.Series]
        returns the used MinMaxScaler object and the converted pd.Series

    """

    # Check if np array, otherwise transform
    if not isinstance(gcm_values, np.ndarray):
        gcm_values = gcm_values.values.reshape(-1, len(gcm_values.columns))

    # If scaler is supplied (e.g. in test values), use this one
    if scaler is not None:
        # Fit Y-values with second MinMaxScaler
        gcm_values[:, 0:1] = scaler[1].transform(gcm_values[:, 0:1])
        # Fit all other variables with first MinMaxScaler
        gcm_values[:, 1:] = scaler[0].transform(gcm_values[:, 1:])

        # Return scaler and values
        return scaler[0], scaler[1], gcm_values

    else:
        # Create scaler object
        scaler_x: MinMaxScaler = MinMaxScaler(feature_range=(min_v, max_v))
        scaler_y: MinMaxScaler = MinMaxScaler(feature_range=(min_v, max_v))

        # Transform GCM values
        gcm_values[:, 0:1] = scaler_y.fit_transform(gcm_values[:, 0:1])
        gcm_values[:, 1:] = scaler_x.fit_transform(gcm_values[:, 1:])

        # Return scaler and gcm_values
        return scaler_x, scaler_y, gcm_values


def scale_gcm_data(gcm_values: pd.Series, scaler: MinMaxScaler = None,
                   min_v: int = 0, max_v: int = 1) -> Tuple[MinMaxScaler, np.array]:
    """
    Scales the GCM data according to `min_v` and `max_v` values.
    Returns the scaler as well as the transformed data.

    Parameters
    ----------
    gcm_values: np.array
        the raw GCM values
    scaler: MinMaxScaler
        if scaler is specified (e.g. in test values) use this object
    min_v: int, optional, standard = 0
        the minimal value of the transformation, standard 0
    max_v: int, optional, standard = 1
        the maximal value of the transformation, standard 1


    Returns
    -------
    Tuple[MinMaxScaler, pd.Series]
        returns the used MinMaxScaler object and the converted pd.Series

    """

    # Check if np array, otherwise transform
    if not isinstance(gcm_values, np.ndarray):
        gcm_values = gcm_values.values.reshape(-1, 1)

    # If scaler is supplied (e.g. in test values), use this one
    if scaler is not None:
        # Fit transform values based on previous scaler
        gcm_values: np.array = scaler.transform(gcm_values)

        # Return scaler and values
        return scaler, gcm_values

    else:
        # Create scaler object
        scaler: MinMaxScaler = MinMaxScaler(feature_range=(min_v, max_v))

        # Transform GCM values
        gcm_values: np.array = scaler.fit_transform(gcm_values)

        # Return scaler and gcm_values
        return scaler, gcm_values


def split_train_validation_test(data: np.array,
                                train_size: float,
                                validation_size: float) -> Tuple[np.array, np.array, np.array]:
    """

    Parameters
    ----------
    data: np.array
        the input data which should be split
    train_size: float
        the relative size of the training dataset (float between 0-1)
    validation_size: float
        the relative size of the validation dataset within the training dataset

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        returns the train, validation and test datasets

    """

    # Type hints variables
    train: np.array
    validation: np.array
    test: np.array

    # Check if np array, otherwise transform
    if not isinstance(data, np.ndarray):
        # NOTE: on 30/9 we changed .reshape(-1, 1) to .reshape(-1, len(cols)) to work with
        # NOTE: multivariate data
        # TODO: check if this works for GCM data also
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif isinstance(data, pd.DataFrame):
            data = data.values.reshape(-1, len(data.columns))

    # Define absolute train and test size
    train_size_abs: int = int(len(data) * train_size)
    validation_size_abs: int = int(train_size_abs * (1 - validation_size))

    # Obtain train & validation dataset
    train = data[0:train_size_abs, :]
    train, validation = train[0:validation_size_abs,
                              :], train[validation_size_abs:len(train), :]

    # Obtain test dataset
    test = data[train_size_abs:len(data), :]

    # Return values
    return train, validation, test
