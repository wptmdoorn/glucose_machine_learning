"""src/glucose/evaluation/clinical/parkes.py

We will evaluate the models with the Parkes
error grid.
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

# Plotting
from methcomp.glucose import parkes
import matplotlib.pyplot as plt

# ML imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model

# Typing
from typing import List, Dict

# Script imports
from glucose.utils.data import split_gcm_datav2, scale_gcm_data, split_train_validation_test

import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# Directories (hard-coded for now)
DATASET_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"
MODEL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\models\\glucose\\"
MODEL_FILE: str = '20200303-142317_BasicRNN_glucose_MODEL.hdf5'
SCALER_FILE: str = '20200303-142317_BasicRNN_glucose_SCALER.pkl'
TEST_FILE: str = '20200303-133436_ACTIVPALGCM_test.pkl'
RAW_DIR: str = 'E:\\Projecten\\Glucose voorspelling\\results\\raw\\'

# DATA CONSTANTS
LOOK_BACK = 6
LOOK_FORWARD = [3, 12]  # 15, 30 mins, 60 mins, 120 mins and 240 mins
PRED_HORIZONS = {'15min': 3, '60min': 12}
PRED_HORIZONS_L = ['15min', '60min']

GROUPS = {'NORMAL': 0.0,
          'PRED': 1.0,
          'T2D': 2.0,
          'Unknown': 3.0}
GROUPS_LIST = ['Normal', 'PreD', 'T2DM', 'Unknown']
status = pd.read_excel(r'E:\Projecten\Glucose voorspelling\data\static\20190924_Glucose status.xlsx',
                       index_col=0).to_dict()['VAL']


logging.info('Start with script')
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

raw_dict = {}

test_df = pickle.load(open(os.path.join(DATASET_DIR, TEST_FILE), 'rb'))
scaler = pickle.load(open(os.path.join(MODEL_DIR, SCALER_FILE), 'rb'))
model = load_model(os.path.join(MODEL_DIR, MODEL_FILE))

for test_ind in test_df[0:100]:
    # predict all values of a test individual
    _id = test_ind[0]
    _idstatus = GROUPS_LIST[int(status.get(int(_id), 3))]
    raw_dict[_id] = {}

    # Obtain data and scale according to scaler
    _df = test_ind[1]
    _df = _df[['time', 'glucose']].copy()

    _df['glucose'] = scaler.transform(_df[['glucose']])

    _x, _y = split_gcm_datav2(_df.to_numpy(), LOOK_BACK, LOOK_FORWARD, pred_index=1)

    try:
        _xgluc = _x[:, :, 1].reshape(-1, LOOK_BACK, 1).astype(np.float64)

        predictions = [scaler.inverse_transform(y) for y in model.predict(_xgluc)]

        _, ytime = split_gcm_datav2(_df.to_numpy(), LOOK_BACK, LOOK_FORWARD, pred_index=0)

        for i, u in enumerate(range(len(predictions))):
            _real = scaler.inverse_transform(_y.T[u].reshape(-1, 1))
            _pred = predictions[u]
            _ytime = ytime.T[u].reshape(-1, 1)

            raw_dict[_id][PRED_HORIZONS_L[i]] = pd.DataFrame.from_dict(
                {'time': _ytime.T[0],
                 'real': _real.T[0],
                 'pred': _pred.T[0]
                 })

    except:
        pass

print(raw_dict)
from pathlib import Path
import shutil
path = os.path.join(RAW_DIR, "{}_{}".format(os.path.basename(__file__), cur_date))
Path(path).mkdir(parents=True, exist_ok=True)

logging.info('Copying script file to output directory')
shutil.copy(__file__, os.path.join(path, 'script_file.py'))

for horizon in ['15min', '60min']:
    concat_df = pd.concat([x[horizon] for x in raw_dict.values() if horizon in x])
    concat_df = concat_df.append(pd.DataFrame.from_dict(
        {'time': [np.nan], 'real': [40], 'pred': [40]}
    ))

    parkes(1,
           concat_df['real'],
           concat_df['pred'],
           units='mmol',
           color_points='red',
           grid=False)
    plt.savefig(os.path.join(path, "{}_{}_{}.png".format(
        'Parkes_overview', horizon, 'glucose'
    )), dpi=300)
    print(concat_df.shape)
    plt.close()

    concat_df = pd.concat([x[horizon] for id, x in raw_dict.items() \
                          if int(status.get(int(id), 3)) == 2 and horizon in x])
    concat_df = concat_df.append(pd.DataFrame.from_dict(
        {'time': [np.nan], 'real': [40], 'pred': [40]}
    ))
    print(concat_df.shape)

    parkes(1,
           concat_df['real'],
           concat_df['pred'],
           units='mmol',
           percentage=True)
    plt.savefig(os.path.join(path, "{}_{}_{}.png".format(
        'Parkes_T2DM', horizon, 'glucose'
    )), dpi=300)
    plt.close()
