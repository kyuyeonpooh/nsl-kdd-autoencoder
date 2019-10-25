import os

import pandas as pd
import numpy as np
import h5py

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import shuffle


def get_service_list(dirname='list', filename='service.txt'):
    print('Getting service list...')
    with open(os.path.join(dirname, filename), 'r') as service:
        service_list = service.read().split('\n')
    return service_list


def get_flag_list(dirname='list', filename='flag.txt'):
    print('Getting flag list...')
    with open(os.path.join(dirname, filename), 'r') as flag:
        flag_list = flag.read().split('\n')
    return flag_list


def get_data_frame(dirname='dataset', filename=None):
    if filename is None:
        raise ValueError('File name should be set.')
    print('Getting data frame from a file...')
    print('Reading file:', os.path.join(dirname, filename))
    df = pd.read_csv(os.path.join(dirname, filename), header=None)
    return df


def to_numeric(data_frame, service_list, flag_list, test=False, attack=False, save=True):
    if test and attack:
        raise ValueError('Test data cannot have attack logs.')
    df = data_frame

    if not test:
        if not attack:
            # extract only normal data
            print('Data size before normal data extraction:', df.shape)
            # index 41: label (index number starts from 0)
            df = df[df[41] == 'normal'].copy()
            print('Data size after normal data extraction:', df.shape)
        else:
            # extract 99% normal data and 1% attack data
            print('Data size before normal data extraction:', df.shape)
            # index 41: label (index number starts from 0)
            df_normal_copy = df[df[41] == 'normal'].copy()
            df_normal = df_normal_copy.sample(n=int(df_normal_copy.shape[0] * 0.99), random_state=1398)
            df_attack = df[df[41] != 'normal'].sample(n=int(df_normal_copy.shape[0] * 0.01) + 1, random_state=1398)
            print('Number of normal records:', len(df_normal))
            print('Number of attack records:', len(df_attack))
            df = pd.concat((df_normal, df_attack))
            print('Data size of concatenated data:', df.shape)
        df = shuffle(df, random_state=1398)

    # index 1: protocol_type
    print('Replacing protocol_type values to numeric...')
    df[1].replace(['tcp', 'udp', 'icmp'], range(3), inplace=True)

    # index 2: service
    print('Replacing service values to numeric...')
    df[2].replace(service_list, range(len(service_list)), inplace=True)

    # index 3: flag
    print('Replacing flag values to numeric...')
    df[3].replace(flag_list, range(len(flag_list)), inplace=True)

    if not test:
        # extract only the same features from Kyoto 2006+ dataset
        df = df.loc[:, [0, 1, 2, 3, 4, 5, 22, 24, 25, 28, 31, 32, 35, 37, 38]]
    else:
        # include label
        df = df.loc[:, [0, 1, 2, 3, 4, 5, 22, 24, 25, 28, 31, 32, 35, 37, 38, 41]]
        df[41] = df[41].map(lambda x: 0 if x == 'normal' else 1)  # normal 0, attack 1

    # save as csv file
    if save:
        if not os.path.exists('csv'):
            os.makedirs('csv')
        if not test:
            if not attack:
                print('Saving file:', os.path.join('csv', 'train_normal_numeric.csv'))
                df.to_csv(os.path.join('csv', 'train_normal_numeric.csv'))
            else:
                print('Saving file:', os.path.join('csv', 'train_mixed_numeric.csv'))
                df.to_csv(os.path.join('csv', 'train_mixed_numeric.csv'))
        else:
            print('Saving file:', os.path.join('csv', 'test_numeric.csv'))
            df.to_csv(os.path.join('csv', 'test_numeric.csv'))

    return df


def to_machine_readable(data_frame, service_list, flag_list, test=False, attack=False, save_csv=True):
    if test and attack:
        raise ValueError('Test data cannot have attack logs.')
    df = data_frame
    sc = MinMaxScaler()
    enc = OneHotEncoder(categories=[range(3), range(len(service_list)), range(len(flag_list))])
    num_desc = df.loc[:, [0, 4, 5]].describe()

    # extract and drop label
    label, df_label = [], []
    if test:
        label = df[41].copy().values.reshape((df.shape[0], 1))
        df_label = pd.DataFrame(label)
        df.drop([41], axis=1, inplace=True)

    # index 0, 4, 5: duration, src_bytes, dst_bytes (in kyoto: index 0, 2, 3)
    attr_name = ['duration', '', '', '', 'src_bytes', 'dst_bytes']
    for i in [0, 4, 5]:
        print('Converting {0} data (index {1}) to machine readable...'.format(attr_name[i], i))
        iqr = (num_desc[i].values[6] - num_desc[i].values[4])
        std = num_desc[i].values[6] + iqr * 1.5  # IQR upper fence = Q3 + 1.5 * IQR
        if std == 0:
            df[i] = df[i].map(lambda x: 1 if x > 0 else 0)
        else:
            df[i] = df[i].map(lambda x: std if x > std else x)
    sc.fit(df[[0, 4, 5]].values)
    df[[0, 4, 5]] = sc.transform(df[[0, 4, 5]].values)

    # index 22, 31, 32: count, dst_host_count, dst_host_srv_count (in kyoto: index 4, 8, 9)
    print('Converting count data (index 22, 31, 32) to machine readable...')
    sc.fit(df[[22, 31, 32]].values.astype(np.float32))
    df[[22, 31, 32]] = sc.transform(df[[22, 31, 32]].values.astype(np.float32))

    # index 1, 2, 3: protocol_type, service, flag (in kyoto: index 23, 1, 13)
    print('Converting type data (index 1, 2, 3) to machine readable...')
    enc.fit(df[[1, 2, 3]].values)
    one_hot_arr = enc.transform(df[[1, 2, 3]].values).toarray()

    # drop one-hot data and attach it again
    print('Dropping and attaching one-hot encoding data...')
    df.drop([1, 2, 3], axis=1, inplace=True)
    df_final = np.concatenate((df.values, one_hot_arr), axis=1)
    df_final = pd.DataFrame(df_final)

    # print shape of data frame
    print('Final shape of data:', df_final.shape)
    if test:
        print('Final shape of label', df_label.shape)

    # save data frame into csv format
    if save_csv:
        if not os.path.exists('csv'):
            os.makedirs('csv')
        if not test:
            if not attack:
                print('Saving file:', os.path.join('csv', 'train_normal_final.csv'))
                df_final.to_csv(os.path.join('csv', 'train_normal_final.csv'), index=False)
            else:
                print('Saving file:', os.path.join('csv', 'train_mix_final.csv'))
                df_final.to_csv(os.path.join('csv', 'train_mix_final.csv'), index=False)
        else:
            print('Saving file:', os.path.join('csv', 'test_feature_final.csv'))
            df_final.to_csv(os.path.join('csv', 'test_feature_final.csv'), index=False)
            print('Saving file:', os.path.join('csv', 'test_label_final.csv'))
            df_label.to_csv(os.path.join('csv', 'test_label_final.csv'), index=False)

    # save into hdf5 format
    if not os.path.exists(os.path.join('..', 'hdf5')):
        os.makedirs(os.path.join('..', 'hdf5'))
    if not test:
        if not attack:
            with h5py.File(os.path.join('..', 'hdf5', 'train_normal.hdf5'), 'w') as hdf:
                print('Saving file:', os.path.join('..', 'hdf5', 'train_normal.hdf5'))
                hdf['x'] = df_final.values[:]
        else:
            with h5py.File(os.path.join('..', 'hdf5', 'train_mix.hdf5'), 'w') as hdf:
                print('Saving file:', os.path.join('..', 'hdf5', 'train_mix.hdf5'))
                hdf['x'] = df_final.values[:]
    else:
        with h5py.File(os.path.join('..', 'hdf5', 'test.hdf5'), 'w') as hdf:
            print('Saving file:', os.path.join('..', 'hdf5', 'test.hdf5'))
            hdf['x'] = df_final.values[:]
            hdf['y'] = df_label[:]
