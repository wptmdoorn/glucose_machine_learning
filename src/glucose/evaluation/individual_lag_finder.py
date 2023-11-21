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

from scipy import signal
import numpy as np

def lag_finder(y1, y2, sr):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()


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

results = {g: {'15min': [], '60min': []} for g in GROUPS_LIST}

logging.info('Exporting individual plots')
for ind, df_dict in raw_dict.items():
    for horizon, raw_data in df_dict.items():
        idstatus = raw_data[0]
        raw_data = raw_data[1]

        raw_data['time'] = pd.to_datetime(raw_data['time'])

        print(ind, idstatus, horizon)
        correls = np.correlate(raw_data['real'], raw_data['pred'], mode='full')
        correls /= np.sqrt(np.dot(raw_data['real'], raw_data['real']) * np.dot(raw_data['pred'], raw_data['pred']))
        maxlags = 20
        lags = np.arange(-maxlags, maxlags + 1)
        correls = correls[len(raw_data['real']) - 1 - maxlags:len(raw_data['real']) + maxlags]

        print(np.argmax(correls))
        print(np.where(lags == 0))

        results[idstatus][horizon].append(((np.where(lags == 0)[0] - np.argmax(correls))[0]) * 5)

        #lag_finder(raw_data['real'], raw_data['pred'], 1/300)

for ind, df_dict in results.items():
    print('{} - {} (n={}): {:.3f} (+- {:.3f})'.format(ind, '15min', len(df_dict['15min']),
                                       np.mean(df_dict['15min']),
                                       np.std(df_dict['15min'])))
    print(np.quantile(df_dict['15min'], 0.25, interpolation='midpoint'))
    print(np.quantile(df_dict['15min'], 0.75, interpolation='midpoint'))
    print(df_dict['15min'])
    print('{} - {} (n={}): {:.3f} (+- {:.3f})'.format(ind, '60min', len(df_dict['60min']),
                                       np.mean(df_dict['60min']),
                                       np.std(df_dict['60min'])))
    print(np.quantile(df_dict['60min'], 0.25, interpolation='midpoint'))
    print(np.quantile(df_dict['60min'], 0.75, interpolation='midpoint'))
    print(df_dict['60min'])


total_15 = results['Normal']['15min'] + results['PreD']['15min'] + results['T2DM']['15min']
total_60 = results['Normal']['60min'] + results['PreD']['60min'] + results['T2DM']['60min']

print('{} - {} (n={}): {:.3f} (+- {:.3f})'.format('TOTAL', '15min', len(total_15),
                                                  np.mean(total_15),
                                                  np.std(total_15)))

print('{} - {} (n={}): {:.3f} (+- {:.3f})'.format('TOTAL', '60min', len(total_60),
                                                  np.mean(total_60),
                                                  np.std(total_60)))