
import os
import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import shuffle


def get_service_list(dirname='input', filename='service.txt'):
    print('Getting service list...')
    with open('{0}/{1}'.format(dirname, filename), 'r') as service:
        service_list = service.read().split('\n')
    return service_list


def get_flag_list(dirname='input', filename='flag.txt'):
    print('Getting flag list...')
    with open('{0}/{1}'.format(dirname, filename), 'r') as flag:
        flag_list = flag.read().split('\n')
    return flag_list


def get_data_frame(dirname):
    print('Getting data frame from files...')
    filelist = os.listdir(dirname)
    frames = []
    for filename in filelist:
        print('Preprocessing file: {0}/{1}'.format(dirname, filename))
        df_temp = pd.read_csv('{0}/{1}'.format(dirname, filename), sep=',', header=None)
        frames.append(df_temp)
    df = pd.concat(frames, ignore_index=True)
    return df


def to_numeric(data_frame, service_list, flag_list, test=False, save=True):
    df = data_frame

    if not test:
        # extract only normal data
        print('Data size before normal data extraction: {0}'.format(df.shape))
        # index 41: label (index number starts from 0)
        df = df[df[41] == 'normal'].copy()
        print('Data size after normal data extraction: {0}'.format(df.shape))
    else:
        # train : test = 50 : 50
        df_normal = df[df[41] == 'normal'].copy()
        df_attack = df[df[41] != 'normal'].sample(n=df_normal.shape[0])
        df = pd.concat((df_normal, df_attack))
        df = shuffle(df)

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
        df[41] = df[41].map(lambda x: 1 if x == 'normal' else 0)

    # save as csv file
    if save:
        if not os.path.exists('csv'):
            os.makedirs('csv')
        if not test:
            print('Saving file: csv/data_frame_train_numeric.csv')
            df.to_csv('csv/data_frame_train_numeric.csv', sep=',', index=False)
        else:
            print('Saving file: csv/data_frame_test_numeric.csv')
            df.to_csv('csv/data_frame_test_numeric.csv', sep=',', index=False)

    return df


def to_machine_readable(data_frame, service_list, flag_list, test=False, save=True):
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
        iqr = (num_desc[i].values[6] - num_desc[i].values[4]) * 1.5
        std = num_desc[i].values[5] + iqr
        if std == 0:
            df[i] = df[i].map(lambda x: 1 if x > 0 else 0)
        else:
            df[i] = df[i].map(lambda x: std if x > std else x)
    sc.fit(df[[0, 4, 5]].values)
    df[[0, 4, 5]] = sc.transform(df[[0, 4, 5]].values)

    # index 22, 31, 32: count, dst_host_count, dst_host_src_count (in kyoto: index 4, 8, 9)
    print('Converting count data (index 22, 31, 32) to machine readable...')
    sc.fit(df[[22, 31, 32]].values.astype(np.float32))
    df[[22, 31, 32]] = sc.transform(df[[22, 31, 32]].values.astype(np.float32))

    # index 1, 2, 3: protocol_type, service, flag (in kyoto: index 23, 1, 13)
    print('Converting type data (index 1, 2, 3) to machine readable...')
    enc.fit(df[[1, 2, 3]].values)
    one_hot_arr = enc.transform(df[[1, 2, 3]].values).toarray()

    # drop one-hot data and attach again
    print('Dropping and attaching one-hot encoding data...')
    df.drop([1, 2, 3], axis=1, inplace=True)
    df_final = np.concatenate((df.values, one_hot_arr), axis=1)

    # drop duplicates (deprecated)
    # print('Before dropping duplicates: {0}'.format(df_final.shape))
    df_final = pd.DataFrame(df_final)
    # df_final.drop_duplicates(inplace=True)
    # print('After dropping duplicates: {0}'.format(df_final.values.shape))

    if save:
        if not os.path.exists('csv'):
            os.makedirs('csv')
        print('Final shape of data frame: {0}'.format(df_final.shape))
        if not test:
            print('Saving file: csv/data_frame_train_final.csv')
            df_final.to_csv('csv/data_frame_train_final.csv', sep=',', index=False)
        else:
            print('Final shape of label: {0}'.format(df_label.shape))
            print('Saving file: csv/data_frame_test_final.csv')
            df_final.to_csv('csv/data_frame_test_final.csv', sep=',', index=False)
            print('Saving file: csv/data_frame_test_label.csv')
            df_label.to_csv('csv/data_frame_test_label.csv', sep=',', index=False)

    # save into hdf5 format (mandatory)
    if not os.path.exists('hdf5'):
        os.makedirs('hdf5')
    if not test:
        with h5py.File('hdf5/nsl_kdd_train.hdf5', 'w') as hdf:
            print('Saving file: hdf5/nsl_kdd_train.hdf5')
            hdf['data'] = df_final.values[:]
    else:
        with h5py.File('hdf5/nsl_kdd_test.hdf5', 'w') as hdf:
            print('Saving file: hdf5/nsl_kdd_test.hdf5')
            hdf['data'] = df_final.values[:]
            hdf['label'] = label[:]
