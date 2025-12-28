import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_datasets():
    sfeatures = ['s' + str(i) for i in range(1, 22)]
    col_names = ['unit_no', 'time', 'op1', 'op2', 'op3'] + sfeatures

    data_train = pd.read_csv('./CMAPSSData/train_FD001.txt', sep=r'\s+', header=None, names=col_names)
    data_test = pd.read_csv('./CMAPSSData/test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    RUL_labels = pd.read_csv('./CMAPSSData/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

    # Add RUL to train
    max_times = data_train.groupby('unit_no')['time'].max().reset_index()
    data_train = data_train.merge(max_times, on='unit_no', suffixes=('', '_max'))
    data_train['RUL_calc'] = data_train['time_max'] - data_train['time']
    data_train.drop(['time', 'time_max'], axis=1, inplace=True)

    # Add RUL to test
    RUL_labels['unit_no'] = RUL_labels.index + 1
    max_times_test = data_test.groupby('unit_no')['time'].max().reset_index()
    data_test = data_test.merge(max_times_test, on='unit_no', suffixes=('', '_max'))
    data_test = data_test.merge(RUL_labels, on='unit_no')
    data_test['RUL_calc'] = data_test['time_max'] - data_test['time'] + data_test['RUL']
    data_test.drop(['time', 'time_max', 'RUL'], axis=1, inplace=True)

    feature_cols = ['op1', 'op2', 'op3'] + sfeatures

    return data_train, data_test, feature_cols


def normalize_data(data_train, data_test, feature_cols):
    scaler = StandardScaler()
    data_train[feature_cols] = scaler.fit_transform(data_train[feature_cols])
    data_test[feature_cols] = scaler.transform(data_test[feature_cols])
    return data_train, data_test