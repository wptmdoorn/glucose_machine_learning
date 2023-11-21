"""src/glucose/development/train_lstm_glucose.py

This builds a LSTM model based upon the
`models.BasicLSTM` class and then saves it.
It uses the pre constructed train and test data from only the
historic glucose data.
"""

# Random seed (important!)
RANDOM_SEED = 1106
import numpy as np
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.compat.v1.set_random_seed(RANDOM_SEED)

# Data science imports
import pandas as pd
import json
import os
import datetime
import pickle

# ML imports
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model

# Script imports
from glucose.utils.data import split_gcm_datav2, scale_gcm_data, split_train_validation_test
from glucose.models.RNN import BasicRNN
from glucose.models.model import DATATYPE

import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# All constants in one directory for easy exports
CONSTANTS = {
    'DATASET_DIR': "E:\\Projecten\\Glucose voorspelling\\data\\final\\",
    'TRAIN_FILE': '20200227-171826_GCM_train.pkl',
    'VAL_FILE': '20200227-171826_GCM_tune.pkl',

    'LOOK_BACK': 6,
    'LOOK_FORWARD': [3, 12],
    'PRED_HORIZONS': {'15min': 3, '60min': 12},

    'DATA_COLS': ['Glucose'],
    'PREDICTION_COL': 'Glucose',

    'EPOCHS': 200,
    'BATCH_SIZE': 32 ** 2,
    'LOSS_FUNC': {'15min': 'mse', '60min': 'mse'},
    'LOSS_WEIGHT': {'15min': 1.0, '60min': 0.1},
    'EARLYSTOPPER_PATIENCE': 10,

    'VERBOSE_LEVEL': 2,
    'NETWORK_CLASS': BasicRNN,
    'DATA_TYPE': 'glucose',
}

MODEL_DIR: str = f"E:\\Projecten\\Glucose voorspelling\\models\\{CONSTANTS['DATA_TYPE']}\\"

# Start of script
logging.info('Start of processing')

# Get cur date
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

logging.info('Loading train and validation datasets')
train = pickle.load(open(os.path.join(CONSTANTS['DATASET_DIR'],
                                      CONSTANTS['TRAIN_FILE']), 'rb'))
validation = pickle.load(open(os.path.join(CONSTANTS['DATASET_DIR'],
                                           CONSTANTS['VAL_FILE']), 'rb'))

logging.info('Dataset sizes: train [{}] - validation [{}]'.format(
    len(train), len(validation),
))

logging.info('Creating scaler with train data')
full_train_df = pd.concat([d for _, d in train])
main_scaler = MinMaxScaler(feature_range=(0, 1))
main_scaler.fit(full_train_df[CONSTANTS['DATA_COLS']])
del full_train_df

xdata = {}
ydata = {}

logging.info('Reading individual GCM values and scaling them to [0, 1]')
for n, d in zip(['train', 'validation'], [train, validation]):
    xdata[n] = []
    ydata[n] = []

    # loop through individual files
    for p in d:
        _id = p[0]
        _df = p[1]
        _df[CONSTANTS['DATA_COLS']] = main_scaler.transform(_df[CONSTANTS['DATA_COLS']])

        _x, _y = split_gcm_datav2(_df.to_numpy(),
                                  CONSTANTS['LOOK_BACK'],
                                  CONSTANTS['LOOK_FORWARD'],
                                  pred_index=list(_df.columns).index(CONSTANTS['PREDICTION_COL']))

        for i in range(len(_x)):
            xdata[n].append(_x[i])
            ydata[n].append(_y[i])

    xdata[n] = np.array(xdata[n])
    ydata[n] = np.array(ydata[n])

logging.info('Individual GCM value dataset length:')
logging.info('Train [{}] - Validation [{}]'.format(
    len(xdata['train']), len(xdata['validation'])
))

# Export scaler before training
pickle.dump(main_scaler,
            open(os.path.join(MODEL_DIR, "{}_{}_{}_{}.pkl".format(
                cur_date,
                CONSTANTS['NETWORK_CLASS'].__name__,
                CONSTANTS['DATA_TYPE'],
                'SCALER')), 'wb'))

# these are in format for training
transposed_trainy = ydata['train'].T
transposed_validationy = ydata['validation'].T

# Create model
logging.info('Initializing model')
model = CONSTANTS['NETWORK_CLASS'](main_scaler, CONSTANTS['LOOK_BACK'], DATATYPE.GCM)
model.init_network(CONSTANTS['PRED_HORIZONS'])

# Print model summary
logging.info('Printing model summary')
model.get_model().summary(print_fn=logging.info)
plot_model(model.get_model(),
           show_shapes=True,
           to_file=os.path.join(MODEL_DIR, "{}_{}_{}_{}.png".format(
               cur_date,
               CONSTANTS['NETWORK_CLASS'].__name__,
               CONSTANTS['DATA_TYPE'],
               'MODEL_PLOT')))

# Compile model
logging.info('Compiling model')
model.compile(loss=CONSTANTS['LOSS_FUNC'],
              loss_weights=CONSTANTS['LOSS_WEIGHT'])

# Fit model to train data
logging.info('Fitting train data to model')

# Transpose the fit and out data to make sure they are in the right format
fit_data = xdata['train'][:, :, 1].reshape(-1, CONSTANTS['LOOK_BACK'], 1).astype(np.float64)
out_data = [transposed_trainy[0].reshape(-1, 1),
            transposed_trainy[1].reshape(-1, 1)]

# Transpose the fit and out data to make sure they are in the right format
fit_data_val = xdata['validation'][:, :, 1].reshape(-1, CONSTANTS['LOOK_BACK'], 1).astype(np.float64)
out_data_val = [transposed_validationy[0].reshape(-1, 1),
                transposed_validationy[1].reshape(-1, 1)]

# Initialize all callbacks
earlystopper = EarlyStopping(monitor='val_loss',
                             patience=CONSTANTS['EARLYSTOPPER_PATIENCE'], verbose=False)
logdir = os.path.join("..\\graph\\", cur_date)
checkdir = os.path.join(MODEL_DIR, '{}_{}_{}_{}.hdf5'.format(
    cur_date,
    CONSTANTS['NETWORK_CLASS'].__name__,
    CONSTANTS['DATA_TYPE'],
    'MODEL',
))

tb = TensorBoard(log_dir=logdir, histogram_freq=2, profile_batch=0,
                 write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(checkdir,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

# Actual fitting
model.fit(fit_data, [out_data[0], out_data[1]],
          epochs=CONSTANTS['EPOCHS'],
          batch_size=CONSTANTS['BATCH_SIZE'],
          verbose=CONSTANTS['VERBOSE_LEVEL'],
          shuffle=True,
          validation_data=(fit_data_val, out_data_val),
          callbacks=[earlystopper, tb, checkpoint])

logging.info('Finished training')
logging.info('Exporting training information')

with open(os.path.join(MODEL_DIR, '{}_{}_{}_{}.json'.format(
        cur_date,
        CONSTANTS['NETWORK_CLASS'].__name__,
        CONSTANTS['DATA_TYPE'],
        'MODEL_INFO')), 'w') as fp:
    json.dump(str(CONSTANTS), fp)
