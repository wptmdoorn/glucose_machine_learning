from glucose.utils.ohio import get_data, load_ohio_dataset
from glucose.utils.data import split_gcm_datav2
import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import datetime

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


def calc_zones(y_true, y_pred, units='mmol'):
    if units == 'mmol':
        y_true = y_true * 18
        y_pred = y_pred * 18

    _zones = []
    data = np.loadtxt('seg.csv')
    _zones = np.array([data.T[int(p), int(t)] for p, t in zip(y_pred, y_true)])
    _zones_sub = [[] for _ in range(8)]

    edges = list(np.arange(0, 4.5, 0.5))

    for x in range(len(edges)-1):
        _zones_sub[x] = np.array(_zones[(_zones >= edges[x]) & (_zones < edges[x+1])])

    print(len(_zones))
    print([len(x) for x in _zones_sub])

    return [(len(x) / len(_zones)) * 100 for x in _zones_sub]



# SEG PLOT
def plot_seg(y_true, y_pred):
    data = np.loadtxt('seg.csv')
    filtered_arr = gaussian_filter(data, sigma=np.std(data.T), order=0)

    fig, ax = plt.subplots()
    ax.set_xlabel('Actual glucose concentration (mmol/L)')
    ax.set_ylabel('Predicted glucose concentration (mmol/L)')

    plt.xlim([0, 600])
    plt.ylim([0, 600])
    plt.xticks([x * 18 for x in [0, 5, 10, 15, 20, 25, 30]])
    plt.yticks([x * 18 for x in [0, 5, 10, 15, 20, 25, 30]])
    ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30])
    ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])
    colors_list = [(0, 165 / 256, 0),
                   (0, 255 / 256, 0),
                   (255 / 256, 255 / 256, 0),
                   (255 / 256, 0, 0),
                   (128 / 256, 0, 0)]
    nodes = [0.0, 0.4375, 1.0625, 2.7500, 4.000]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", list(zip([x / 4 for x in nodes], colors_list)),
                                                               gamma=1)

    cax = ax.imshow(np.flipud(np.array(plt.imread('seg600.png'))),
                    origin='lower',
                    cmap=cmap,
                    vmin=0,
                    vmax=4)

    cbar = fig.colorbar(cax, ticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                        orientation='vertical',
                        fraction=0.15,
                        aspect=6)
    cbar.ax.tick_params(labelsize=8)
    #cbar.ax.yaxis.set_tick_params(verticalalignment='center')

    seps = [0, 0.5, 1.5, 2.5, 3.5, 4]
    labs = [(0.25, 'None'), (1, 'Slight'), (2.0, 'Moderate'), (3.0, 'High'), (3.75, 'Extreme')]

    for s in seps:
    #   cbar.ax.text(6, s, '|', ha='left', va='center', rotation=90)
        cbar.ax.plot([6, 6.5], [s] * 2, '-', color='black', lw=1, alpha=1, clip_on=False)

    trans = cbar.ax.get_xaxis_transform()

    #for s in range(len(seps) - 1):
    #    print(seps[s], seps[s+1])
    #    cbar.ax.plot([6, 6], [seps[s]+.03, seps[s+1]-.03], '-', clip_on=False, color='black', lw=1)


    for l in labs:
        cbar.ax.text(6.2, l[0]-.008, l[1], ha='left', va='center', rotation=0, fontsize=10)

    zones = calc_zones(y_true, y_pred)

    for i, x in enumerate(zones):
        cbar.ax.plot([0, 5], [(i * .5) + .5] * 2, '--', color='grey', lw=1, alpha=.6)

        if x > 0:
            if round(x, 2) == 0:
                _str = "<0.01%"
            else:
                _str = "{:.2f}%".format(x)

            cbar.ax.text(2, (i * 0.5) + .25, _str,
                         ha='center', va='center', fontsize=9)

    y_true = y_true * 18
    y_pred = y_pred * 18

    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.set_ylabel('Risk score')

    plt.scatter(y_true, y_pred, s=8,
                edgecolors='black',
                facecolors='white',
                alpha=.8)

# DATA CONSTANTS
LOOK_BACK = 6
LOOK_FORWARD = [3, 6, 12]  # 15, 30 mins, 60 mins, 120 mins and 240 mins
PRED_HORIZONS = {'15min': 3, '30min': 6, '60min': 12}
PRED_HORIZONS_L = ['15min', '30min', '60min']

print(datasets)

scaler = pickle.load(open(os.path.join(MODEL_DIR, SCALER_FILE), 'rb'))
model = load_model(os.path.join(MODEL_DIR, MODEL_FILE))
raw_dict = {}

for test_ind in datasets:
    # predict all values of a test individual
    _id = test_ind[0]
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

        raw_dict[_id][PRED_HORIZONS_L[i]] = pd.DataFrame.from_dict(
            {'time': _ytime.T[0],
             'real': _real.T[0],
             'pred': _pred.T[0]
             }
        )
        _r = math.sqrt(mean_squared_error(
            _real, _pred,
        ))

        print(f'Patient: {_id} - Horizon: {PRED_HORIZONS_L[i]} - RMSE: {_r * 18}')


from pathlib import Path
import shutil
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
path = os.path.join(RAW_DIR, "{}_{}".format(os.path.basename(__file__), cur_date))
Path(path).mkdir(parents=True, exist_ok=True)

shutil.copy(__file__, os.path.join(path, 'script_file.py'))

for horizon in ['15min', '30min', '60min']:
    concat_df = pd.concat([x[horizon] for id, x in raw_dict.items()])
    #concat_df = concat_df.append(pd.DataFrame.from_dict(
    #    {'time': [np.nan], 'real': [40], 'pred': [40]}
    #))
    print(concat_df.shape)

    concat_df['pred'] = concat_df['pred'] + ((concat_df['real'] - concat_df['pred']) * 0.04)

    over_pred = [p > t for p, t in zip(concat_df['pred'], concat_df['real'])]

    print('P>T: total, N= {} - perc={}'.format(sum(over_pred),
                                               (sum(over_pred)/len(concat_df))*100))

    plot_seg(concat_df['real'], concat_df['pred'])

    plt.savefig(os.path.join(path, "{}_{}_{}.png".format(
        'OHIO_SEG_T1DM', horizon, 'glucose'
    )), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
