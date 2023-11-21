""""src/glucose/process/split_data_gcmactivpal.py

Generate split data-sets for training, hyperparameter tuning and
testing datasets.
"""

# general imports
import pickle
import pandas as pd
import os
import datetime

# script imports
from glucose.utils.files import list_files
from glucose.settings import SEED

# Set random seed
import random
random.seed(SEED)

# Setup logger
import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# CONSTANTS FOR SPLITTING
TRAIN_SPLIT = 0.7
TUNE_SPLIT = 0.1
TEST_SPLIT = 0.2  # set the None if we only want to use train and tune splits
OUT_DIR = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"
IN_ACTIVPAL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\activpal_gcm\\"

# Starting script
logging.info('Listing files in GCM-ActivPal directory:')
logging.info(IN_ACTIVPAL_DIR)

activpal_files = list_files(IN_ACTIVPAL_DIR, extension=".xlsx", recursive=True)

# List all files of specific subgroup
sfiles = [i for i in activpal_files]
logging.info('Found {} files of ActivPal-GCM data'.format(len(sfiles)))

# Read all the files and parse into datasets object
logging.info('Reading all data into an internal object')
datasets = []

for c, i in enumerate(sfiles):
    if c % 10 == 0:
        logging.info('Proccessed {}th file'.format(c))

    # raw dataframe
    _d = pd.read_excel(i,
                       names=['time', 'glucose', 'x', 'y', 'z'],
                       sheetIndex=0)

    # first we sum the X, Y and Z values
    _newd = _d.groupby(_d.index // 20).sum()

    # next we add the glucose and time values
    _newd.iloc[:, 0] = _d.iloc[::20, 1].values
    _newd['time'] = _d['time'][::20].values

    # append
    datasets.append((i.split('.')[0].split('\\')[-1], _newd))

random.shuffle(datasets)

train_dfs = datasets[0: int(TRAIN_SPLIT * len(datasets))]
tune_dfs = datasets[int(TRAIN_SPLIT * len(datasets)): int((TRAIN_SPLIT + TUNE_SPLIT) * len(datasets))]
test_dfs = datasets[int((TRAIN_SPLIT + TUNE_SPLIT) * len(datasets)):]

logging.info('Dataset sizes: ')
logging.info('Train [{}] - Tune [{}] - Test [{}]'.format(
    len(train_dfs), len(tune_dfs), len(test_dfs)
))

logging.info('Exporting data to directory: ')
logging.info(OUT_DIR)

cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

for name, df in [('train', train_dfs), ('tune', tune_dfs), ('test', test_dfs)]:
    logging.info('Concatting and exporting {} dataset'.format(name))

    with open(os.path.join(OUT_DIR, '{}_ACTIVPALGCM_{}.pkl'.format(
            cur_time, name)),
              'wb') as f:
        pickle.dump(df, f)

logging.info('Finished exporting')
