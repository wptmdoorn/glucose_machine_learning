"""
glucose/paper/tune_hyperparams.py

This script is used to select the best model
and tune hyparparameters. We will use the train
and validation datasets.
"""

# general imports
import datetime
import pickle
import os
import numpy as np
import pandas as pd
from glucose.settings import SEED

# hyperparameter tuning imports
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.utils import shuffle

# machine learning model building
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, SimpleRNN, Bidirectional, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler

# script imports
from glucose.utils.data import scale_gcm_data, split_gcm_datav2

import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")


DATASET_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"
MODEL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\models\\"
TRAIN_FILE: str = '20200227-171826_GCM_train.pkl'
VAL_FILE: str = '20200227-171826_GCM_tune.pkl'
IN_DIR = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"
LOOK_BACK = 12
LOOK_FORWARD = [3, 12]
ITERATIONS = 100
GLOB_I = 0

SEARCH_SPACE = {
    # layer and units 1
    'dense1': hp.choice('dense1', ['GRU', 'RNN', 'LSTM', 'BI_RNN', 'BI_LSTM']),
    'units1': hp.choice('units1', [4, 8, 16, 32, 64, 128]),
    'activation1': hp.choice('activation1', ['relu', 'sigmoid', 'tanh']),

    # layer and units 2
    'dense2': hp.choice('dense2', [None, 'GRU', 'RNN', 'LSTM', 'BI_RNN', 'BI_LSTM']),
    'units2': hp.choice('units2', [4, 8, 16, 32, 64, 128]),
    'activation2': hp.choice('activation2', ['relu', 'sigmoid', 'tanh']),

    # layer and units 3
    'units3': hp.choice('units3', [2, 6, 8, 10, 18, 20, 24, 28]),
    'activation3': hp.choice('activation3', ['relu', 'sigmoid', 'tanh']),

    # training / compiling parameters
    'batch_size': hp.choice('batch_size', [4096, 2048, 1024, 512, 256, 128, 64]),
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop', 'SGD']),
    # 'learning_rate': hp.uniform('learning_rate', 1, 6)
}


def get_data_and_scaler(model: str):
    if model == 'gcm':
        logging.info('Loading train and validation datasets')
        train = pickle.load(open(os.path.join(DATASET_DIR, TRAIN_FILE), 'rb'))
        validation = pickle.load(
            open(os.path.join(DATASET_DIR, VAL_FILE), 'rb'))

        logging.info('Dataset sizes: train [{}] - validation [{}]'.format(
            len(train), len(validation),
        ))

        logging.info('Creating scaler with train data')
        full_train_df = pd.concat([d for _, d in train])
        main_scaler = MinMaxScaler(feature_range=(0, 1))
        main_scaler.fit(full_train_df[['Glucose']])
        del full_train_df

        xdata = {}
        ydata = {}

        logging.info(
            'Reading individual GCM values and scaling them to [0, 1]')
        for n, d in zip(['train', 'validation'], [train, validation]):
            xdata[n] = []
            ydata[n] = []

            # loop through individual files
            for p in d:
                _id = p[0]
                _df = p[1]
                _df['Glucose'] = main_scaler.transform(_df[['Glucose']])

                _x, _y = split_gcm_datav2(
                    _df.to_numpy(), LOOK_BACK, LOOK_FORWARD, pred_index=1)

                for i in range(len(_x)):
                    xdata[n].append(_x[i])
                    ydata[n].append(_y[i])

            xdata[n] = np.array(xdata[n])
            ydata[n] = np.array(ydata[n])

        logging.info('Individual GCM value dataset length:')
        logging.info('Train [{}] - Validation [{}]'.format(
            len(xdata['train']), len(xdata['validation'])
        ))

        # these are in format for training
        transposed_trainy = ydata['train'].T
        transposed_validationy = ydata['validation'].T

        # Transpose the fit and out data to make sure they are in the right format
        fit_data = xdata['train'][:, :,
                                  1].reshape(-1, LOOK_BACK, 1).astype(np.float64)
        out_data = [transposed_trainy[0].reshape(-1, 1),
                    transposed_trainy[1].reshape(-1, 1)]

        # Transpose the fit and out data to make sure they are in the right format
        fit_data_val = xdata['validation'][:, :,
                                           1].reshape(-1, LOOK_BACK, 1).astype(np.float64)
        out_data_val = [transposed_validationy[0].reshape(-1, 1),
                        transposed_validationy[1].reshape(-1, 1)]

        return main_scaler, fit_data, out_data, fit_data_val, out_data_val


def objective(params):
    global GLOB_I
    _opt_str_to_class = {'Adam': Adam, 'Nadam': Nadam,
                         'RMSprop': RMSprop, 'SGD': SGD}

    print('[{}/{}] New iteration initialized with:\n {}'.format(GLOB_I +
          1, ITERATIONS, params))
    input1 = Input(batch_shape=(None, LOOK_BACK, 1))

    # First layer
    if params['dense1'] == 'RNN':
        layer1 = SimpleRNN(params['units1'],
                           return_sequences=params['dense2'] is not None,
                           activation=params['activation1'],
                           input_shape=(LOOK_BACK, 1))(input1)
    elif params['dense1'] == 'LSTM':
        layer1 = LSTM(params['units2'],
                      activation=params['activation1'],
                      return_sequences=params['dense2'] is not None,
                      input_shape=(LOOK_BACK, 1))(input1)
    elif params['dense1'] == 'BI_RNN':
        layer1 = Bidirectional(SimpleRNN(params['units1'],
                                         activation=params['activation1'],
                                         return_sequences=params['dense2'] is not None,
                                         input_shape=(LOOK_BACK, 1)))(input1)
    elif params['dense1'] == 'BI_LSTM':
        layer1 = Bidirectional(LSTM(params['units1'],
                                    activation=params['activation1'],
                                    return_sequences=params['dense2'] is not None,
                                    input_shape=(LOOK_BACK, 1)))(input1)

    # Possible second (and third) layer
    if params['dense2'] is not None:
        if params['dense2'] == 'RNN':
            layer2 = SimpleRNN(
                params['units2'], activation=params['activation2'])(layer1)
        elif params['dense2'] == 'LSTM':
            layer2 = LSTM(params['units2'],
                          activation=params['activation2'])(layer1)
        elif params['dense2'] == 'BI_RNN':
            layer2 = Bidirectional(
                SimpleRNN(params['units2'], activation=params['activation2']))(layer1)
        elif params['dense2'] == 'BI_LSTM':
            layer2 = Bidirectional(
                LSTM(params['units2'], activation=params['activation2']))(layer1)

        layer3 = Dense(params['units3'],
                       activation=params['activation3'])(layer2)

    else:
        layer3 = Dense(params['units3'],
                       activation=params['activation2'])(layer1)

    outs = [Dense(1, name="15min")(layer3),
            Dense(1, name="60min")(layer3)]

    model = Model(inputs=input1, outputs=outs)

    # Define losses for each of the predictions
    losses = {"15min": "mse",
              "60min": "mse"}

    loss_w = {"15min": 1.0,
              "60min": 1.0}

    model.compile(loss=losses,
                  loss_weights=loss_w,
                  metrics=['accuracy', RootMeanSquaredError(name='rmse')],
                  optimizer=_opt_str_to_class[params['optimizer']](
                      lr=0.01, decay=0.001,  # LETS FIX THE LEARNING RATE
                  ))

    result = model.fit(x_train, y_train,
                       batch_size=int(params['batch_size']),
                       epochs=10,
                       verbose=0,
                       validation_data=(x_tune, y_tune))

    # Get the highest validation accuracy of the training epochs
    validation_acc = np.amin(result.history['val_loss'])
    logging.info('[{}/{}] Mean Loss: {:.4f} | Best Loss: {:.4f} (iteration: {})'.format(
        GLOB_I+1,
        ITERATIONS,
        np.mean(result.history['val_loss']),
        validation_acc,
        result.history['val_loss'].index(validation_acc)
    ))

    GLOB_I += 1

    return {'loss': validation_acc,
            'status': STATUS_OK,
            'model_space': params}


value_scaler, x_train, y_train, x_tune, y_tune = get_data_and_scaler('gcm')

trials = Trials()
best = fmin(objective,
            SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=ITERATIONS,
            trials=trials)

# Save the results
MODEL_DIR = "E:\\Projecten\\Glucose voorspelling\\models\\"
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

with open(os.path.join(MODEL_DIR, '{}_GCM_TUNING_HISTORY.pkl'.format(cur_date)), 'wb') as f:
    pickle.dump({'best': best,
                 'history': trials}, f)
