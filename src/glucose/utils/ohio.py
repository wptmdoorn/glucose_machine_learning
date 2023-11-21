import pandas as pd
import xml.etree.ElementTree as et
import io
import os
import numpy as np
import datetime
from dateutil.parser import parse

def iter_element(etree, tag):
    i=0
    for element in etree.find(tag):
        i += 1
        for key, value in element.attrib.items():
            yield i, key, value

def compute_interval(df, threshold=5, interval_name='ts'):
    # Timestamp interval bins in minutes
    threshold = 5
    threshold_ns = threshold * 60 * 1e9

    df['interval'] = pd.to_datetime(np.round(df[interval_name].astype(np.int64) / threshold_ns) * threshold_ns)

    return df

def calculate_duration(x,y):
    if x != '' and y != '':
        if y>x:
            duration = y-x
        else:
            duration = x-y
        return str(datetime.timedelta(seconds=int(duration.total_seconds())))
    return ''


def get_data(path, tag, interval):
    etree = et.parse(path).getroot()
    df = pd.DataFrame(iter_element(etree, tag))
    df = df.pivot(columns=1, index=0)
    df.columns = df.columns.levels[1]
    if 'ts' in df.columns:
        df.ts = df.ts.apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H:%M:%S'))
        df = compute_interval(df, interval, 'ts')
    elif 'ts_begin' in df.columns:
        df['ts_begin'] = df['ts_begin'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H:%M:%S') if x != '' else x)
        df['ts_end'] = df['ts_end'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H:%M:%S') if x != '' else x)
        df['duration'] = df.apply(lambda x: calculate_duration(x['ts_begin'], x['ts_end']), axis=1)
        df = compute_interval(df, interval, 'ts_begin')
    elif 'tbegin' in df.columns:
        df['tbegin'] = df['tbegin'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H:%M:%S') if x != '' else x)
        df['tend'] = df['tend'].apply(lambda x: datetime.datetime.strptime(x, '%d-%m-%Y %H:%M:%S') if x != '' else x)
        df['duration'] = df.apply(lambda x: calculate_duration(x['tbegin'], x['tend']), axis=1)
        df = compute_interval(df, interval, 'tbegin')
    return df


def load_ohio_dataset(base_path):
    ohio_data_files = os.listdir(base_path)
    datasets = []

    for f in ohio_data_files:
        # get data
        df = get_data(os.path.join(base_path, f), 'glucose_level', 5)

        # only obtain timestamp and glucose value
        df = df[['ts', 'value']]
        df = df.rename(columns={'ts': 'Timestamp', 'value': 'Glucose'})
        df['Glucose'] = df['Glucose'].astype(float) * 0.0555

        # append float with patient ID
        datasets.append((f, df))

    return datasets