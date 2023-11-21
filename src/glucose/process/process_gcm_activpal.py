"""src/glucose/process/process_gcm_activpal.py

This script contains everything what is needed to preprocess
the GCM and ActivPal data. It reads both the GCM and ActivPal files,
integrates them by linear interpolation of the GCM data - and finally
exports them to a database. Please see --help and the inline comments
for additional documentation.

"""

# General imports
import argparse
import os
from tqdm import tqdm

# Data science imports
import pandas as pd

# Script specific imports
from glucose.utils.files import list_files
from glucose.utils.processing import activpal_time_to_date, add_years, nearest_item
from glucose.static.activpal import INCLUSION_VARIABLES

# Typing imports
from typing import List, Union

# Date time imports
from datetime import timedelta

# Logger
import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# Random seed (important!)
RANDOM_SEED = 1106
import numpy as np
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)

# Static variables
IN_GCM_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\gcm\\final\\"
IN_ACTIVPAL_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\activpal\\15sec_VANE\\"
OUT_DATA_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\activpal_gcm\\"
MEAS_TO_DROP: int = 0


# noinspection PyBroadException
def parse_single_activpal(file_path: str) -> Union[str, pd.DataFrame]:
    """

    Parameters
    ----------
    file_path: str
        this variable contains the full file path to the ActivPal data

    Returns
    -------
    Union[None, pd.DataFrame]
        returns either None or a pd.DataFrame depending on whether it could find something or not

    """

    try:
        # Parse patient ID and check for available GCM file
        _patient_id: int = int(file_path.split('\\')[-1].split('-')[0])

    except Exception:
        # If not able to cast to an int (e.g. invalid file name), then
        # Return none and proceed to next file
        return 'Invalid file name - abort.'

    # Check how many GCM files are available
    _gcm_file: List[str] = list_files(IN_GCM_DIR,
                                      extension="{}.csv".format(_patient_id))

    # Check length of GCM files
    if len(_gcm_file) == 0:
        # If no files are found, return and proceed to next ActivPal file
        return 'No GCM files found - abort.'

    if len(_gcm_file) > 1:
        return 'Found multiple files for patient {}, namely: {}'.format(_patient_id, _gcm_file)

    # NOTE: this was added on 01/10/2019 to prevent the script from re-processing
    # NOTE: about 600 files. if we adjust the processing code someday, we should remove
    # NOTE: these lines!
    if os.path.exists(os.path.join(OUT_DATA_DIR, "{}.xlsx".format(_patient_id))):
        return 'File already exists in output directory - abort.'

    # Read ActivPal and GCM data data
    _activpal_data: pd.DataFrame = pd.read_csv(file_path, skiprows=[1])
    _gcm_data: pd.DataFrame = pd.read_csv(_gcm_file[0], parse_dates=[0])

    # ActivPal parsing
    # First remove unnecessary fields
    _activpal_data = _activpal_data[INCLUSION_VARIABLES]

    # Parse time field and add 1900 years
    _activpal_data['time'] = _activpal_data['time'].apply(activpal_time_to_date)
    _activpal_data['time'] = _activpal_data['time'].apply(add_years, args= (1900,))

    # Round all ActivPal data to 15second date intervals
    # This is to ensure that e.g. 00:00:14 and 00:00:16 also get recognized
    _activpal_data['time'] = _activpal_data['time'].dt.round('15s')

    # Rename column for compatibility purposes with the GCM data
    _activpal_data = _activpal_data.rename(columns={'time': 'Timestamp'}).set_index('Timestamp')

    # GCM parsing
    _gcm_data = _gcm_data.set_index('Timestamp')

    # Obtain first and last values for GCM and ActivPal
    _activpal_start = _activpal_data.head(1).index
    _gcm_start = _gcm_data.head(1).index
    _activpal_end = _activpal_data.tail(1).index
    _gcm_end = _gcm_data.tail(1).index

    if abs((_gcm_start - _activpal_start).days) > 1:
        return "Difference between GCM and ActivPal too big - abort."

    # Create 15-second data frame for GCM data
    oidx = _gcm_data.index
    nidx = pd.date_range(oidx.min(), oidx.max(), freq='15s')
    _gcm_15s = _gcm_data.reindex(oidx.union(nidx)).interpolate('time').reindex(nidx)

    # Calculate nearest ActivPal data
    _activpal_nearest = nearest_item(_activpal_data.index, _gcm_start)
    _diff_seconds = (_activpal_nearest - _gcm_start).total_seconds()

    # Synchronize the 15S GCM frame to this!
    _gcm_15s.index = _gcm_15s.index + timedelta(seconds=_diff_seconds[0])

    return pd.merge(_gcm_15s, _activpal_data, left_index=True, right_index=True)


def setup_arguments() -> argparse.ArgumentParser:
    """Parses the arguments for this file.

    Parameters
    ----------
    None.

    Returns
    -------
    ArgumentParser
        Returns newly instanced ArgumentParser object with all parameter options.

    """

    # import and setup ArgumentParser object
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    return ap


def main() -> None:
    logging.info('Starting data processing')

    # Reading arguments
    logging.info('Retrieving file arguments')
    args: argparse.Namespace = setup_arguments().parse_args()

    # List all files to read
    logging.info('Listing all raw ActivPal files')
    activpal_files: List[str] = list_files(IN_ACTIVPAL_DIR, extension=".csv", recursive=True)
    logging.info('Found {} raw ActivPal files'.format(len(activpal_files)))

    # Obtain visits table
    visit_table = pd.read_excel(os.path.join('..', 'data', 'static', 'id_visits.xlsx'))
    visit_table_v7 = visit_table.dropna(subset=['V7'])

    # Iterating through dataframes and export a minimized version of them
    i: int = 0
    logging.info('Starting processing individual ActivPal profiles')

    for i in tqdm(range(0, len(activpal_files))):
        # Obtain unique ID
        try:
            uid = int(activpal_files[i].split('\\')[-1].split('-')[0])

        except ValueError:
            logging.error("[{}/{} ({})] Invalid ID - abort.".format(i, len(activpal_files), uid))
            continue

        # Check visits
        if not any(visit_table['Patient ID'] == uid):
            logging.error("[{}/{} ({})] ID not found in visits - abort.".format(i, len(activpal_files),
                                                                                uid))
            continue

        # Check visits for baseline-followup
        if any(visit_table_v7['Patient ID'] == uid):
            logging.error("[{}/{} ({})] ID has V7 visit - no processing of data".format(i, len(activpal_files),
                                                                                        uid))
            continue

        # Read file
        processed_activpal: Union[str, pd.DataFrame] = parse_single_activpal(activpal_files[i])

        # An error occurred if the return value is a str
        if isinstance(processed_activpal, str):
            logging.error("[{}/{} ({})] {}".format(i, len(activpal_files),
                                                   uid, processed_activpal))
            continue

        processed_activpal.to_excel(os.path.join(OUT_DATA_DIR,
                                                 "{}.xlsx".format(uid)))
        logging.info("[{}/{} ({})] Successfully processed file.".format(i, len(activpal_files),
                                                                        uid))

        i += 1

    pass

if __name__ == "__main__":
    main()
