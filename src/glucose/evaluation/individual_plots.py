"""src/glucose/evaluation/individual_plots.py

Generate individual plots for glucose based predictions
of all individuals.
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
from tensorflow.keras.models import load_model

# Script imports
from glucose.utils.data import split_gcm_datav2
from glucose.utils.evaluation import r_squared

import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# Directories (hard-coded for now)
DATASET_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"
MODEL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\models\\glucose\\"
MODEL_FILE: str = '20200303-142317_BasicRNN_glucose_MODEL.hdf5'
SCALER_FILE: str = '20200303-142317_BasicRNN_glucose_SCALER.pkl'
TEST_FILE: str = '20200227-171826_GCM_test.pkl'
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

for test_ind in test_df:
    # predict all values of a test individual
    _id = test_ind[0]
    _idstatus = GROUPS_LIST[int(status.get(int(_id), 3))]
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
        _ytime = ytime.T[u].reshape(-1, 1)

        raw_dict[_id][PRED_HORIZONS_L[i]] = (_idstatus, pd.DataFrame.from_dict(
            {'time': _ytime.T[0],
             'real': _real.T[0],
             'pred': _pred.T[0]
             }
        ))


from pathlib import Path
import shutil
path = os.path.join(RAW_DIR, "{}_{}".format(os.path.basename(__file__), cur_date))
Path(path).mkdir(parents=True, exist_ok=True)

logging.info('Copying script file to output directory')
shutil.copy(__file__, os.path.join(path, 'script_file.py'))

logging.info('Exporting individual plots')
for ind, df_dict in raw_dict.items():
    for horizon, raw_data in df_dict.items():
        idstatus = raw_data[0]
        raw_data = raw_data[1]

        raw_data['time'] = pd.to_datetime(raw_data['time'])
        rsquared = r_squared(raw_data['pred'], raw_data['real'])

        # Plotting
        plt.figure()
        if horizon ==  '15min':
            plt.plot(raw_data['time'], raw_data['real'], label='Actual', alpha=0.5, linewidth=0.6)
            plt.plot(raw_data['time'], raw_data['pred'], label='Prediction', alpha=0.5, linewidth=0.6)
        else:
            plt.plot(raw_data['time'], raw_data['real'], label='Actual', alpha=0.5, linewidth=0.6)
            plt.plot(raw_data['time'] - pd.Timedelta(minutes=30), raw_data['pred'], label='Prediction', alpha=0.5, linewidth=0.6)

        #plt.title('ID: {} ({}) - {} - R2 of {}'.format(ind, idstatus, horizon, rsquared))
        plt.ylabel('Glucose concentration (mmol/L)')
        plt.ylim([0, 20])
        plt.gca().set_xticklabels([f'Day {x}' for x in range(1, 7)])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.margins(x=0)
        plt.legend(frameon=False)
        plt.savefig(os.path.join(path, "{}_{}_{}_{:.2f}.png".format(
            ind, idstatus, horizon, rsquared)), dpi=1200)
        plt.close()
