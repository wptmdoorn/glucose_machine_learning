"""
src/glucose/paper/transfer_learning_01.py

Transfer-learning using the OhioT1DM dataset.
"""

# Data science imports
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# ML imports
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, SimpleRNN, Bidirectional, Dense, Input
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Typing
from typing import List, Dict

# Script imports
from utils.data import split_gcm_datav2, scale_gcm_data, split_train_validation_test

# Random seed (important!)
RANDOM_SEED = 1106
np.random.seed(RANDOM_SEED)
tf.compat.v1.set_random_seed(RANDOM_SEED)

GCM_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\gcm\\final\\"
LOOK_BACK = 12
# 15, 30 mins, 60 mins, 120 mins and 240 mins
LOOK_FORWARD = [3, 6, 12, 24, 48]
GROUPS = {'NORMAL': 0.0,
          'PRED': 1.0,
          'T2D': 2.0}
TEST_SIZE = 0.2


def get_models() -> Dict:
    """
    Returns a dictionary of all models with its
    key the name and its Model instance as item.

    Returns
    -------
    Dict
        dictionary containing the model-name and model instance
    """

    # Instantiate empty Dictionary
    _models: Dict = {}

    # LSTM Model
    input1 = Input(batch_shape=(None, LOOK_BACK, 1))
    lay1 = LSTM(64, input_shape=(LOOK_BACK, 1))(input1)
    lay2 = Dense(8)(lay1)
    outs = [Dense(1, name="one")(lay2), Dense(1, name="two")(lay2),
            Dense(1, name="three")(lay2), Dense(1, name="four")(lay2), Dense(1, name="five")(lay2)]
    _models["LSTM"] = Model(inputs=input1, outputs=outs)

    return _models


# filter files only to contain normal-glucose
gcm_files: List[str] = os.listdir(GCM_DIR)
status = pd.read_excel(
    r'E:\Projecten\Glucose voorspelling\data\static\20190924_Glucose status.xlsx')
main_results = {}

# List all files of specific subgroup
sfiles = [i for i in gcm_files if any(
    status[status['ID'] == float(i.split('.csv')[0])]['VAL'] == 0.0)]
print('Found {} files of specific subgroup'.format(len(sfiles)))

datasets = []
main_scaler = None

# read some FILES
for i in sfiles:
    _d = pd.read_csv(os.path.join(GCM_DIR, i))
    _d = _d['Glucose']

    # Scale data, otherwise use previous scaler
    if main_scaler is None:
        main_scaler, _d = scale_gcm_data(_d)
    else:
        _, _d = scale_gcm_data(_d, scaler=main_scaler)
    datasets.append(_d)

# Generate a random array with zeros and ones
train_test_mask = np.array([0] * int(TEST_SIZE * len(sfiles) + 1) + [1] * int((1 - TEST_SIZE) * len(sfiles)),
                           dtype=np.bool)
np.random.shuffle(train_test_mask)

# Convert datasets to array and subset
datasets = np.array(datasets)
train = datasets[train_test_mask]
test = datasets[~train_test_mask]

train_x = []
train_y = []
for d in datasets:
    _x, _y = split_gcm_datav2(d, LOOK_BACK, LOOK_FORWARD)
    for i in range(len(_x)):
        train_x.append(_x[i])
        train_y.append(_y[i])

train_x, train_y = shuffle(np.array(train_x), np.array(
    train_y), random_state=RANDOM_SEED)
tylist = [[i[n] for i in train_y] for n in range(0, 5)]

model = get_models()['LSTM']

# Define losses for each of the predictions
losses = {"one": "mse",
          "two": "mse",
          "three": "mse",
          "four": "mse",
          "five": "mse"}

# Define loss weights
lossW = {"one": 1.0, "two": 1.0, "three": 1.0, "four": 1.0, "five": 1.0}

model_name = 'LSTM'
model_instance = model

optimizer: Optimizer = Adam(lr=0.001, decay=0.005)
model_instance.compile(loss=losses,
                       loss_weights=lossW,
                       optimizer=optimizer)

# Create earlystopper object
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=False)

# Start fitting the data to the model
print("* Training {}".format(model_name))
model_instance.fit(train_x, tylist,
                   validation_split=0.2,
                   callbacks=[earlystopper],
                   verbose=2, batch_size=131072,
                   epochs=5)

for patient in list(test):
    _x, _y = split_gcm_datav2(patient, LOOK_BACK, LOOK_FORWARD)

    print(len(_x))
