from glucose.utils.ohio import get_data, load_ohio_dataset
from glucose.utils.data import split_gcm_datav2
import os
import pickle
import itertools
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Directories (hard-coded for now)
MODEL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\models\\glucose"
MODEL_FILE: str = '20201126-102247_BasicLSTM_glucose_MODEL.hdf5'
SCALER_FILE: str = '20201126-102247_BasicLSTM_glucose_SCALER.pkl'
TEST_FILE: str = '20200227-171826_GCM_test.pkl'
RAW_DIR: str = 'E:\\Projecten\\Glucose voorspelling\\results\\raw\\'
OHIO_TEST_PATH = r'E:\Projecten\Glucose voorspelling\data\ohiot1dm\2020\test'
OHIO_TEST_PATH_FILES = os.listdir(OHIO_TEST_PATH)
datasets = load_ohio_dataset(r'E:\Projecten\Glucose voorspelling\data\ohiot1dm\2020\test')
OUT_PATH = r'E:\Projecten\Glucose voorspelling\results\indiv_plots_ohio'

# DATA CONSTANTS
LOOK_BACK = 6
LOOK_FORWARD = [3, 12]  # 15, 30 mins, 60 mins, 120 mins and 240 mins
PRED_HORIZONS = {'15min': 3, '60min': 12}
PRED_HORIZONS_L = ['15min', '60min']

print(datasets)

scaler = pickle.load(open(os.path.join(MODEL_DIR, SCALER_FILE), 'rb'))
model = load_model(os.path.join(MODEL_DIR, MODEL_FILE))

for year, dataset in itertools.product([2018, 2020], ['train', 'test']):
    _path = rf'E:\Projecten\Glucose voorspelling\data\ohiot1dm\{year}\{dataset}'
    datasets = load_ohio_dataset(_path)

    for test_ind in datasets:
        # predict all values of a test individual
        _id = test_ind[0]

        # Obtain data and scale according to scaler
        _df = test_ind[1]

        plt.figure()
        plt.plot(_df   ['Timestamp'], _df['Glucose'], label='Actual', alpha=0.5)

        # plt.title('ID: {} ({}) - {} - R2 of {}'.format(ind, idstatus, horizon, rsquared))
        plt.ylabel('Glucose concentration (mmol/L)')
        plt.xticks(rotation=45)
        plt.ylim([0, 40])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.margins(x=0)
        plt.legend(frameon=False)
        plt.title(f'{_id} ({dataset} - {year})')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_PATH, "{}_{}_{}.png".format(year, dataset, _id)))
        plt.close()
