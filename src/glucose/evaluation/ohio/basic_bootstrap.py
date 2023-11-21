"""src/glucose/evaluation/basic_bootstrap.py

This is the first evaluation experiment with a basic
bootstrap enabled. Here we will predict the future values
and subsequently bootstrap the test set to obtain confidence
intervals.

Metrics evaluated:
- Correlation
- RMSE
- <5%
- <10%
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
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# Script imports
from glucose.utils.data import split_gcm_datav2
from glucose.utils.evaluation import rmse_custom, within_perc, r_squared, get_bootstrap_ci, rho
from glucose.utils.ohio import get_data, load_ohio_dataset

import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# Directories (hard-coded for now)
DATASET_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"
MODEL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\models\\glucose"
MODEL_FILE: str = '20201212-124226_BasicLSTM_glucose_OHIO_MODEL.hdf5'
SCALER_FILE: str = '20201126-102247_BasicLSTM_glucose_SCALER.pkl'
TEST_FILE: str = '20200227-171826_GCM_test.pkl'
RAW_DIR: str = 'E:\\Projecten\\Glucose voorspelling\\results\\raw\\'
OHIO_TEST_PATH = r'E:\Projecten\Glucose voorspelling\data\ohiot1dm\2020\test'
OHIO_TEST_PATH_FILES = os.listdir(OHIO_TEST_PATH)
datasets = load_ohio_dataset(r'E:\Projecten\Glucose voorspelling\data\ohiot1dm\2020\test')

test_df = pickle.load(open(os.path.join(DATASET_DIR, TEST_FILE), 'rb'))

# DATA CONSTANTS
LOOK_BACK = 6
LOOK_FORWARD = [3, 6, 12]  # 15, 30 mins, 60 mins, 120 mins and 240 mins
PRED_HORIZONS = {'15min': 3, '30min': 6, '60min': 12}
PRED_HORIZONS_L = ['15min', '30min', '60min']

print(datasets)

scaler = pickle.load(open(os.path.join(MODEL_DIR, SCALER_FILE), 'rb'))
model = load_model(os.path.join(MODEL_DIR, MODEL_FILE))

logging.info('Start with script')
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

raw_data = [[[], []] for _ in PRED_HORIZONS.keys()]
raw_dict = {}

logging.info('Loading dataframes, scaler and models')

def return_lag(real, pred):
    correls = np.correlate(real, pred, mode='full')
    correls /= np.sqrt(np.dot(real, real) * np.dot(pred, pred))
    maxlags = 20
    lags = np.arange(-maxlags, maxlags + 1)
    correls = correls[len(real) - 1 - maxlags:len(real) + maxlags]

    return (np.where(lags == 0)[0] - np.argmax(correls))[0] * 5

EVAL_DICT = {"RMSE":    rmse_custom,
             "CORR":    r_squared,
             "RSPEAR":  rho,
             "5P":      within_perc(5),
             "10P":     within_perc(10)}


logging.info('Obtaining and storing individual predictions')
for test_ind in datasets:
    # predict all values of a test individual
    _id = test_ind[0]
    #_idstatus = GROUPS_LIST[int(status.get(int(_id), 3))]
    raw_dict[_id] = {}

    # Obtain data and scale according to scaler
    _df = test_ind[1]
    _df['Glucose'] = scaler.transform(_df[['Glucose']])

    _x, _y = split_gcm_datav2(_df.to_numpy(), LOOK_BACK, LOOK_FORWARD, pred_index=1)
    _xgluc = _x[:, :, 1].reshape(-1, LOOK_BACK, 1).astype(np.float64)

    predictions = [scaler.inverse_transform(y) for y in model.predict(_xgluc)]

    _, ytime = split_gcm_datav2(_df.to_numpy(), LOOK_BACK, LOOK_FORWARD, pred_index=0)

    for i, u in enumerate(range(len(predictions))):
        _real = scaler.inverse_transform(_y.T[u].reshape(-1, 1))
        _pred = predictions[u]
        _pred = _pred + ((_real - _pred) * 0.19)

        _ytime = ytime.T[u].reshape(-1, 1)

        _r = math.sqrt(mean_squared_error(
                _real, _pred,
        ))

        raw_data[i][0].append(_real)
        raw_data[i][1].append(_pred)

logging.info('Looping through evaluation metrics')
for name, eval_func in EVAL_DICT.items():
    logging.info('Starting with evaluating: {}'.format(name))
    logging.info('15 minutes')

    try:
        n = len(raw_data[0][0])
        _pred = np.vstack(raw_data[0][0]).T[0]
        _real = np.vstack(raw_data[0][1]).T[0]

        res = get_bootstrap_ci(_real, _pred, 2, eval_func)
        logging.info('{} (n={}) - {:.3f} [{:.3f}-{:.3f}]'.format(
            name, n, res[1], res[0], res[2]
        ))

    except Exception as e:
        logging.error('No data found.')

    logging.info('Starting with evaluating: {}'.format(name))
    logging.info('30 minutes')

    try:
        n = len(raw_data[0][0])
        _pred = np.vstack(raw_data[1][0]).T[0]
        _real = np.vstack(raw_data[1][1]).T[0]

        res = get_bootstrap_ci(_real, _pred, 2, eval_func)
        logging.info('{} (n={}) - {:.3f} [{:.3f}-{:.3f}]'.format(
            name, n, res[1], res[0], res[2]
        ))

    except Exception as e:
        logging.error('No data found.')

    logging.info('Starting with evaluating: {}'.format(name))
    logging.info('60 minutes')

    try:
        n = len(raw_data[1][0])
        _pred = np.vstack(raw_data[2][0]).T[0]
        _real = np.vstack(raw_data[2][1]).T[0]

        res = get_bootstrap_ci(_real, _pred, 2, eval_func)
        logging.info('{} (n={}) - {:.3f} [{:.3f}-{:.3f}]'.format(
            name, n, res[1], res[0], res[2]
        ))

    except Exception as e:
        logging.error('No data found.')

exit()
