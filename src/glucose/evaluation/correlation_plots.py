"""src/glucose/evaluation/correlation_plots.py

Plot simple plots for each group for 15- and 60-minute
time intervals.
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
import matplotlib.pyplot as plt

# ML imports
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# Script imports
from glucose.utils.data import split_gcm_datav2
from glucose.utils.evaluation import r_squared

import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# Directories (hard-coded for now)
DATASET_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"
MODEL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\models\\"
MODEL_FILE: str = '20200228-093104_LSTM_GLUC_MODEL.hdf5'
SCALER_FILE: str = '20200228-093104_LSTM_GLUC_SCALER.pkl'
TEST_FILE: str = '20200227-171826_GCM_test.pkl'
RAW_DIR: str = 'E:\\Projecten\\Glucose voorspelling\\results\\raw\\'

# DATA CONSTANTS
LOOK_BACK = 12
LOOK_FORWARD = [3, 12]  # 15, 30 mins, 60 mins, 120 mins and 240 mins
PRED_HORIZONS = {'15min': 3, '60min': 12}
PRED_HORIZONS_L = ['15min', '60min']

GROUPS = {'NORMAL': 0.0,
          'PRED': 1.0,
          'T2D': 2.0}
GROUPS_LIST = ['Normal', 'PreD', 'T2DM']
status = pd.read_excel(r'E:\Projecten\Glucose voorspelling\data\static\20190924_Glucose status.xlsx',
                       index_col=0).to_dict()['VAL']


logging.info('Start with script')
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

raw_data = {x: [[[], []] for _ in PRED_HORIZONS.keys()] for x in GROUPS_LIST}

logging.info('Loading dataframes, scaler and models')
test_df = pickle.load(open(os.path.join(DATASET_DIR, TEST_FILE), 'rb'))
scaler = pickle.load(open(os.path.join(MODEL_DIR, SCALER_FILE), 'rb'))
model = load_model(os.path.join(MODEL_DIR, MODEL_FILE))

logging.info('Obtaining and storing individual predictions')
for test_ind in test_df[0:20]:
    # predict all values of a test individual
    _id = test_ind[0]
    _idstatus = GROUPS_LIST[int(status.get(int(_id), 3))]

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
        _ytime = ytime.T[u].reshape(-1, 1)

        _r = math.sqrt(mean_squared_error(
                _real, _pred,
        ))

        raw_data[_idstatus][i][0].append(_real)
        raw_data[_idstatus][i][1].append(_pred)


logging.info('Looping through evaluation metrics')
for horizon in [0, 1]:
    logging.info('Starting with correlation')
    logging.info('15 minutes')

    for g in GROUPS_LIST:
        _pred = np.vstack(raw_data[g][0][0]).T[0]
        _real = np.vstack(raw_data[g][0][1]).T[0]

        # Plot individual points
        plt.scatter(_pred, _real, s=6, alpha=0.6)

        # Limits
        plt.xlim([0, 25])
        plt.ylim([0, 25])

        # Titles
        plt.title('{} ({}) - R2 of {}'.format(
            g, PRED_HORIZONS_L[horizon], r_squared(_pred, _real)
        ))
        plt.show()

exit()

from pathlib import Path
import shutil
path = os.path.join(RAW_DIR, cur_date)
Path(path).mkdir(parents=True, exist_ok=True)

logging.info('Copying script file to output directory')
shutil.copy(__file__, os.path.join(path, 'script_file.py'))

logging.info('Saving all raw data')
for ind, df_dict in raw_dict.items():
    writer = pd.ExcelWriter(os.path.join(path, '{}.xlsx'.format(ind)), engine='xlsxwriter')

    for horizon, raw_data in df_dict.items():
        raw_data.to_excel(writer, sheet_name=horizon)

    writer.save()
