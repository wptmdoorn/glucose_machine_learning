""""
glucose/paper/split_data.py

Generate split data-sets for training, hyperparameter tuning and
testing datasets.
"""

# general imports
import pickle
import pandas as pd
import os

# script imports
from glucose.utils.files import list_files

import random
SEED = 1106
random.seed(SEED)

# SETTINGS
GLUCOSE_STATUS_SPLIT = False  # split on glucose status or not?
TRAIN_SPLIT = 0.7
TUNE_SPLIT = 0.2
TEST_SPLIT = 0.1  # set the None if we only want to use train and tune splits
OUT_DIR = "E:\\Projecten\\Glucose voorspelling\\data\\final\\"


# PART 1:
# GCM DATA
print('Start with splitting of GCM data')
GCM_DIR: str = r"E:\\Projecten\\Glucose voorspelling\\data\\final\\gcm\\"
gcm_files = list_files(GCM_DIR, extension='*.csv')

if GLUCOSE_STATUS_SPLIT:
    pass

else:
    list_of_df = [pd.read_csv(f) for f in gcm_files]
    print('Found {} GCM-files'.format(len(list_of_df)))
    random.shuffle(list_of_df)

    train_dfs = list_of_df[0: int(TRAIN_SPLIT * len(list_of_df))]
    tune_dfs = list_of_df[int(TRAIN_SPLIT * len(list_of_df))                          : int((TRAIN_SPLIT + TUNE_SPLIT) * len(list_of_df))]
    test_dfs = list_of_df[int((TRAIN_SPLIT + TUNE_SPLIT) * len(list_of_df)):]

    print('Dataset sizes')
    print('* Train: {} \n* Tune : {} \n* Test: {}\n'.format(len(train_dfs),
                                                            len(tune_dfs),
                                                            len(test_dfs)))

    for name, df in [('train', train_dfs), ('tune', tune_dfs), ('test', test_dfs)]:
        print('Concatting and exporting {} dataset'.format(name))
        concatted_df = pd.concat(df)
        with open(os.path.join(OUT_DIR, 'GCM_{}.pkl'.format(name)), 'wb') as f:
            pickle.dump(concatted_df, f)

print('\n')

# PART 2:
# ACTIVPAL AND GCM DATA
print('Start with splitting of ActivPal and GCM data')
IN_ACTIVPAL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\activpal_gcm\\"
activpal_files = list_files(IN_ACTIVPAL_DIR, extension=".xlsx", recursive=True)

if GLUCOSE_STATUS_SPLIT:
    pass

else:
    list_of_df = [pd.read_excel(f, sheetIndex=0) for f in activpal_files]
    print('Found {} ActivPal+GCM-files'.format(len(list_of_df)))
    random.shuffle(list_of_df)

    train_dfs = list_of_df[0: int(TRAIN_SPLIT * len(list_of_df))]
    tune_dfs = list_of_df[int(TRAIN_SPLIT * len(list_of_df))                          : int((TRAIN_SPLIT + TUNE_SPLIT) * len(list_of_df))]
    test_dfs = list_of_df[int((TRAIN_SPLIT + TUNE_SPLIT) * len(list_of_df)):]

    print('Dataset sizes')
    print('* Train: {} \n* Tune : {} \n* Test: {}\n'.format(len(train_dfs),
                                                            len(tune_dfs),
                                                            len(test_dfs)))

    for name, df in [('train', train_dfs), ('tune', tune_dfs), ('test', test_dfs)]:
        print('Concatting and exporting {} dataset'.format(name))
        concatted_df = pd.concat(df)

        with open(os.path.join(OUT_DIR, 'ACTIVPAL_GCM_{}.pkl'.format(name)), 'wb') as f:
            pickle.dump(concatted_df, f)
