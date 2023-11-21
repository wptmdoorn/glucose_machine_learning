"""src/glucose/development/train_rnn_glucactivpal.py

This builds a RNN model based upon the
`models.BasicRNN` class and then saves it.
It uses the pre constructed train and test data from the
historic glucose and ActivPal data.
"""

# Random seed (important!)
RANDOM_SEED = 1106
import numpy as np
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.compat.v1.set_random_seed(RANDOM_SEED)

# Data science imports
import pandas as pd
import math
import os
import datetime
import pickle

# ML imports
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

# Typing
from typing import List, Dict

# Script imports
from glucose.utils.data import split_gcm_datav2, scale_gcm_data, split_train_validation_test
from glucose.models.RNN import BasicRNN
from glucose.models.model import DATATYPE

import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# Directories (hard-coded for now)
DATASET_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"
MODEL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\models\\"
TRAIN_FILE: str = '20200302-102830_ACTIVPALGCM_train.pkl'
VAL_FILE: str = '20200302-102830_ACTIVPALGCM_tune.pkl'

# DATA CONSTANTS
LOOK_BACK = 12 * 20  # activpal
LOOK_FORWARD = [3 * 20, 12 * 20]  # 15, 30 mins, 60 mins, 120 mins and 240 mins
PRED_HORIZONS = {'15min': 3 * 20,
                 '60min': 12 * 20}
# TRAIN CONSTANTS
EPOCHS = 200
BATCH_SIZE = 32 ** 3
VERBOSE_LEVEL = 2

# Start of script
logging.info('Start of processing')

# Get cur date
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

logging.info('Loading train and validation datasets')
train = pickle.load(open(os.path.join(DATASET_DIR, TRAIN_FILE), 'rb'))
validation = pickle.load(open(os.path.join(DATASET_DIR, VAL_FILE), 'rb'))


logging.info('Dataset sizes: train [{}] - validation [{}]'.format(
    len(train), len(validation),
))


logging.info('Creating scaler with train data')
full_train_df = pd.concat([d for _, d in train])
data_cols = full_train_df.columns[1:]
main_scaler = MinMaxScaler(feature_range=(0, 1))
main_scaler.fit(full_train_df[full_train_df.columns[1:]])
del full_train_df

xdata = {}
ydata = {}

logging.info('Reading individual GCM values')
# testing purposes
train = train[0:20]
validation = validation[0:5]

for n, d in zip(['train', 'validation'], [train, validation]):
    xdata[n] = []
    ydata[n] = []

    # loop through individual files
    for c, p in enumerate(d):
        if c % 10 == 0:
            logging.info('Processing {}th dataframe of {}'.format(c, n))

        _id = p[0]
        _df = p[1]
        _df[data_cols] = main_scaler.transform(_df[data_cols])

        _x, _y = split_gcm_datav2(_df.to_numpy(), LOOK_BACK, LOOK_FORWARD, pred_index=1)

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
                cur_date, 'RNN', 'GLUC_ACTIVPAL', 'SCALER')), 'wb'))

logging.info('Transposing Y values')
# these are in format for training
transposed_trainy = ydata['train'].T
transposed_validationy = ydata['validation'].T

# Create model
logging.info('Initializing model')
model = BasicRNN(main_scaler, LOOK_BACK, DATATYPE.GCM)
model.init_network(PRED_HORIZONS)

# Print model summary
logging.info('Printing model summary')
model.get_model().summary(print_fn=logging.info)

# Compile model
logging.info('Compiling model')
model.compile(loss={x: 'mse' for x in PRED_HORIZONS.keys()},
              loss_weights={'15min': 1.0, '60min': 0.1})

# Fit model to train data
logging.info('Fitting train data to model')

# Transpose the fit and out data to make sure they are in the right format
fit_data = xdata['train'][:, :, 1].reshape(-1, LOOK_BACK, 1).astype(np.float64)
out_data = [transposed_trainy[0].reshape(-1, 1),
            transposed_trainy[1].reshape(-1, 1)]

# Transpose the fit and out data to make sure they are in the right format
fit_data_val = xdata['validation'][:, :, 1].reshape(-1, LOOK_BACK, 1).astype(np.float64)
out_data_val = [transposed_validationy[0].reshape(-1, 1),
                transposed_validationy[1].reshape(-1, 1)]

# Initialize all callbacks
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=False)
logdir = os.path.join("..\\graph\\", cur_date)
checkdir = os.path.join(MODEL_DIR, '{}_{}_{}_{}.hdf5'.format(
    cur_date, 'RNN', 'GLUC_ACTIVPAL', 'MODEL'
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
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          verbose=VERBOSE_LEVEL,
          shuffle=True,
          validation_data=(fit_data_val, out_data_val),
          callbacks=[earlystopper, tb, checkpoint])

logging.info('Finished training')
