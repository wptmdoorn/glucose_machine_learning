"""src/process/process_gcm.py

This script contains everything what is needed to preprocess
the GCM data. It reads all the files and produces a matrix to
show what is possible using the data. See --help for an overview
of all available options.

"""

# General imports
import argparse

# Data science imports
import pandas as pd

# Script specific imports
from glucose.utils.files import list_files

# Typing imports
from typing import List

# Logger
import logging
import logging.config
logging.config.fileConfig("../logging/logging.conf.txt")

# Static variables
IN_DATA_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\gcm\\"
OUT_DATA_DIR: str = "E:\\Projecten\\Glucose voorspelling\\data\\gcm\\final\\"
MEAS_TO_DROP: int = 12 * 24


# noinspection PyBroadException
def parse_single_gcm(file_path: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    file_path: str
        this variable contains the full file path to the GCM data

    Returns
    -------
    pd.DataFrame
        modified pd.DataFrame

    """

    # Read data
    try:
        _data: pd.DataFrame = pd.read_excel(file_path,
                                            skiprows=11, decimal=",", usecols=[3, 8])
        # Drop NaN values
        _data = _data.dropna(subset=['Sensor Glucose (mmol/L)'])

        # Rename glucose column
        _data.columns: List[str] = ['Timestamp', 'Glucose']

        # Remove first N rows
        _data = _data.iloc[MEAS_TO_DROP:]

        # Return data
        return _data

    except Exception as e:
        logging.error('An error occurred with file: {}'.format(file_path))
        logging.error(e)


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
    logging.info('Listing all raw GCM files')
    gcm_files: List[str] = list_files(IN_DATA_DIR, extension=".xlsx", recursive=True)
    logging.info('Found {} raw GCM files'.format(len(gcm_files)))

    # Iterating through dataframes and export a minimized version of them
    i: int = 0
    logging.info('Starting processing individual glucose profiles')

    for i in range(0, len(gcm_files)):
        # Read file
        processed_gcm: pd.DataFrame = parse_single_gcm(gcm_files[i])

        # Format out data file path
        out_file: str = OUT_DATA_DIR + gcm_files[i].split('\\')[-1].replace('.xlsx', '.csv')

        # Export CSV
        processed_gcm.to_csv(out_file, header=True, index=False)

        # Log every 100 processed GCM files
        if i % 100 == 0:
            logging.info('Processed {} GCM files'.format(i))

        i += 1

    pass


if __name__ == "__main__":
    main()
