"""src/glucose/paper/biLSTM.py
"""

# Random seed (important!)
from tensorflow.keras import backend as K
import logging.config
import logging
from glucose.models.model import DATATYPE
from glucose.models.LSTM import BiLSTM
from glucose.utils.data import split_gcm_datav2, scale_gcm_data, split_train_validation_test
from typing import List, Dict
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import datetime
import os
import math
import pandas as pd
import tensorflow as tf
import numpy as np
RANDOM_SEED = 1106
np.random.seed(RANDOM_SEED)
tf.compat.v1.set_random_seed(RANDOM_SEED)

# Data science imports

# ML imports

# Typing

# Script imports

logging.config.fileConfig("../logging/logging.conf.txt")

GCM_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\gcm\\final\\"

# DATA CONSTANTS
LOOK_BACK = 12
LOOK_FORWARD = [3, 12]  # 15, 30 mins, 60 mins, 120 mins and 240 mins
PRED_HORIZONS = {'15min': 3, '60min': 12}

# TRAIN CONSTANTS
EPOCHS = 200
BATCH_SIZE = 32 ** 3
VERBOSE_LEVEL = 2

GROUPS = {'NORMAL': 0.0,
          'PRED': 1.0,
          'T2D': 2.0,
          'Unknown': 3.0}
GROUPS_LIST = ['Normal', 'PreD', 'T2DM', 'Unknown']

# filter files only to contain normal-glucose
gcm_files: List[str] = os.listdir(GCM_DIR)
status = pd.read_excel(r'E:\Projecten\Glucose voorspelling\data\static\20190924_Glucose status.xlsx',
                       index_col=0).to_dict()['VAL']

main_results = {}

K.clear_session()
logging.info('Start of processing')

# List all files of specific subgroup
sfiles = [i for i in gcm_files]
logging.info('Found {} files of GCM data'.format(len(sfiles)))

# Read all the files and parse into datasets object
logging.info('Reading all data into an internal object')
datasets = []

for i in sfiles:
    _d = pd.read_csv(os.path.join(GCM_DIR, i))
    datasets.append((i.split('.')[0], _d))

logging.info('Splitting data into train, test and validation datasets')
# Separate data into train, test and validation sets
train, other = train_test_split(datasets,
                                test_size=0.30,
                                random_state=RANDOM_SEED)
validation, test = train_test_split(other,
                                    test_size=0.67,
                                    random_state=RANDOM_SEED)

logging.info('Dataset sizes: train [{}] - validation [{}] - test [{}]'.format(
    len(train), len(validation), len(test)
))

logging.info('Creating scaler with train data')
full_train_df = pd.concat([d for _, d in train])
main_scaler = MinMaxScaler(feature_range=(0, 1))
main_scaler.fit(full_train_df[['Glucose']])
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

# Create model
logging.info('Initializing model')
model = BiLSTM(main_scaler, LOOK_BACK, DATATYPE.GCM)
model.init_network(PRED_HORIZONS)

# Print model summary
logging.info('Printing model summary')
model.get_model().summary(print_fn=logging.info)

# Compile model
logging.info('Compiling model')
model.compile(loss={x: 'mse' for x in PRED_HORIZONS.keys()})

# Fit model to train data
logging.info('Fitting train data to model')

# Transpose the fit and out data to make sure they are in the right format
fit_data = xdata['train'][:, :, 1].reshape(-1, LOOK_BACK, 1).astype(np.float64)
out_data = [transposed_trainy[0].reshape(-1, 1),
            transposed_trainy[1].reshape(-1, 1)]

# Transpose the fit and out data to make sure they are in the right format
fit_data_val = xdata['validation'][:, :,
                                   1].reshape(-1, LOOK_BACK, 1).astype(np.float64)
out_data_val = [transposed_validationy[0].reshape(-1, 1),
                transposed_validationy[1].reshape(-1, 1)]

# Create earlystopper object
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=False)
logdir = os.path.join(
    "..\\graph\\", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=logdir, histogram_freq=2, profile_batch=0,
                 write_graph=True, write_images=True)

# Actual fitting
model.fit(fit_data, [out_data[0], out_data[1]],
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          verbose=VERBOSE_LEVEL,
          shuffle=True,
          validation_data=(fit_data_val, out_data_val),
          callbacks=[earlystopper, tb])

# TODO: pickle model
logging.info('Predicting in test dataset')

rmse = {x: [[] for _ in PRED_HORIZONS.keys()] for x in GROUPS_LIST}

for test_ind in test:
    # predict all values of a test individual
    _id = test_ind[0]
    _idstatus = GROUPS_LIST[int(status.get(int(_id), 3))]

    # Obtain data and scale according to scaler
    _df = test_ind[1]
    _df['Glucose'] = main_scaler.transform(_df[['Glucose']])

    _x, _y = split_gcm_datav2(
        _df.to_numpy(), LOOK_BACK, LOOK_FORWARD, pred_index=1)
    _xgluc = _x[:, :, 1].reshape(-1, LOOK_BACK, 1).astype(np.float64)

    predictions = model.predict(_xgluc)

    for i, u in enumerate(range(len(predictions))):
        _r = math.sqrt(mean_squared_error(
            main_scaler.inverse_transform(_y.T[u].reshape(-1, 1)),
            predictions[u])
        )
        rmse[_idstatus][i].append(_r)

        logging.debug('ID : {} - Type: {} - RMSE: {}'.format(
            _id, _idstatus, _r
        ))


logging.info('Mean RMSE (15min) for each group: ')
logging.info('Normal: {} - PreD: {} - T2DM: {} - Unknown: {}'.format(
    *[np.mean(rmse[x][0]) for x in GROUPS_LIST]
))

logging.info('Mean RMSE (60min) for each group: ')
logging.info('Normal: {} - PreD: {} - T2DM: {} - Unknown: {}'.format(
    *[np.mean(rmse[x][1]) for x in GROUPS_LIST]
))
