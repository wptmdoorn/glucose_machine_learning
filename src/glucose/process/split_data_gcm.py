""""src/glucose/process/split_data_gcm.py

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
GCM_DIR: str = r"E:\\Projecten\\Glucose voorspelling\\data\\final\\gcm\\"
OUT_DIR = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"

# Starting script
logging.info('Listing files in GCM directory:')
logging.info(GCM_DIR)

gcm_files = list_files(GCM_DIR, extension='*.csv')

# List all files of specific subgroup
sfiles = [i for i in gcm_files]
logging.info('Found {} files of GCM data'.format(len(sfiles)))

# Read all the files and parse into datasets object
logging.info('Reading all data into an internal object')
datasets = []

for i in sfiles:
    _d = pd.read_csv(os.path.join(GCM_DIR, i))
    datasets.append((i.split('.')[0].split('\\')[-1], _d))

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

    with open(os.path.join(OUT_DIR, '{}_GCM_{}.pkl'.format(
            cur_time, name)),
              'wb') as f:
        pickle.dump(df, f)

logging.info('Finished exporting')
