from glucose.utils.ohio import get_data, load_ohio_dataset
from glucose.utils.data import split_gcm_datav2
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.models import load_model

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

for test_ind in datasets:
    # predict all values of a test individual
    _id = test_ind[0]

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

        print(f'Patient: {_id} - Horizon: {PRED_HORIZONS_L[i]} - RMSE: {_r * 18}')