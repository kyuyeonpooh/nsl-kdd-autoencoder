#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, LeakyReLU, Dropout
from keras.models import Model, load_model
from keras.utils.io_utils import HDF5Matrix

from sklearn import metrics
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import math


# from AutoencoderIDS import AutoencoderIDS
def myDisance(u, v):
    distance = 0.0
    u = u[0]
    v = v[0]
    for idx in range(u.shape[0]):
        distance += abs(u[idx] - v[idx])

    return distance


def cross_entropy(y, p):
    result = 0
    y = y[0]
    p = p[0]
    for idx in range(y.shape[0]):
        if (y[idx] == 1):
            b = -math.log10(p[idx])
            result += b
        elif (p[idx] == 0):
            print(p[idx], p[idx] == 0)
            result += 0
        else:
            print(p[idx], y[idx])
            b = -(y[idx] * math.log10(p[idx]) + (1 - y[idx]) * math.log10(1 - p[idx]))
            result += b
    return result


def deploy(makeCSV=True, makePlot=True):
    model = load_model('./autoencoder.h5')
    normalDataLoss = []
    attackDataLoss = []

    hdf5_files = os.listdir('./keras_hdf5/test')
    total = 0
    nn = 0
    an = 0
    crt = 0

    dtype = [('error', float), ('is_attack', bool), ('predict', bool)]
    err_list = []

    for hdf5_file in hdf5_files:
        print(hdf5_file)
        data_test = list(HDF5Matrix('./keras_hdf5/test/' + hdf5_file, 'data'))
        label_test = list(HDF5Matrix('./keras_hdf5/test/' + hdf5_file, 'label'))
        total = total + len(data_test)
        for idx in range(len(data_test)):
            input_test = data_test[idx].reshape((1, 114))
            predict_test = model.predict(input_test)
            loss = myDisance(input_test, predict_test)

            atk_flag = False
            if loss > 1.0:
                atk_flag = True
            if label_test[idx] == 1:
                if not atk_flag:
                    crt += 1
                normalDataLoss.append(loss)
                err_list.append((loss, False, atk_flag))
                nn = nn + 1
            else:
                if atk_flag:
                    crt += 1
                attackDataLoss.append(loss)
                err_list.append((loss, True, atk_flag))
                an = an + 1
    print('Accuracy: {0}'.format(crt / (an + nn)))

    err_list = np.array(err_list, dtype=dtype)
    err_low_list = np.copy(err_list)
    err_low_list = np.sort(err_low_list, order='error')
    pd.DataFrame(err_low_list).to_csv('err_list.csv', header=False)
    err_list = np.sort(err_list, order='error')[::-1]
    print(len(err_list))

    rates = [0.0001, 0.0005, 0.001, 0.01, 0.1, 0.15, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50]
    for r in rates:
        top_elements = err_list[:int(len(err_list) * r)]
        precision_real = 0
        precision_pred = 0
        recall_real = 0
        recall_pred = 0
        for i in range(len(top_elements)):
            if top_elements[i][2]:
                precision_pred += 1
                if top_elements[i][1]:
                    precision_real += 1
            if top_elements[i][1]:
                recall_real += 1
                if top_elements[i][2]:
                    recall_pred += 1
        precision = precision_real / precision_pred
        recall = recall_pred / recall_real
        print("{0},{1},{2},{3},{4},{5},{6}".format(r * 100, len(top_elements), recall_real, len(top_elements) - recall_real,
                                               precision,  recall, top_elements[-1][0]))

    allDataLoss = normalDataLoss + attackDataLoss

    rates = [0.0001, 0.0005, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    for r in rates:
        top_elements = err_low_list[:int(len(err_low_list) * r)]
        precision_real = 0
        precision_pred = 0
        recall_real = 0
        recall_pred = 0
        for i in range(len(top_elements)):
            if not top_elements[i][2]:
                precision_pred += 1
                if not top_elements[i][1]:
                    precision_real += 1
            if not top_elements[i][1]:
                recall_real += 1
                if not top_elements[i][2]:
                    recall_pred += 1
        precision = precision_real / precision_pred
        recall = recall_pred / recall_real
        print("{0},{1},{2},{3},{4},{5},{6}".format(r * 100, len(top_elements), recall_real,
                                                   len(top_elements) - recall_real,
                                                   precision, recall, top_elements[-1][0]))

    print("Sum : ", len(allDataLoss))
    allLabel = [1] * len(normalDataLoss) + [0] * len(attackDataLoss)

    fpr, tpr, thresholds = metrics.roc_curve(np.array(allLabel), np.array(allDataLoss), pos_label=0,
                                             drop_intermediate=False)
    print(metrics.auc(fpr, tpr))
    # print(metrics.confusion_matrix())
    print("fpr", fpr)
    print("tpr", tpr)
    print("thresholds", thresholds)
    print("")
    if makePlot:
        if not os.path.exists('./plot'):
            os.makedirs('./plot')
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, )
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('DDOS detection AutoEncoder ROC-curve')
        plt.legend(loc="lower right")
        plt.grid(True)  # add grid
        plt.savefig('./plot/deploy_roc_curve.png', dpi=80)
        plt.show()

    if makeCSV:
        if not os.path.exists('./csv'):
            os.makedirs('./csv')

        truePositiveRate = []
        falsePositiveRate = []
        threshold = []
        recall = []
        precision = []
        specificity = []
        f1_measure = []
        accuracy = []
        for rate in range(10, 21, 1):
            truePositiveRate.append(tpr[np.where(tpr >= (rate * 0.05))[0][0]])
            falsePositiveRate.append(fpr[np.where(tpr >= (rate * 0.05))[0][0]])
            recall.append(truePositiveRate[-1])
            precision.append((truePositiveRate[-1] * len(attackDataLoss)) / (
                        truePositiveRate[-1] * len(attackDataLoss) + falsePositiveRate[-1] * len(normalDataLoss)))
            specificity.append(1 - falsePositiveRate[-1])
            f1_measure.append((2 * recall[-1] * precision[-1]) / (precision[-1] + recall[-1]))
            threshold.append(thresholds[np.where(tpr >= (rate * 0.05))[0][0]])
            accuracy.append((truePositiveRate[-1] * an + (1 - falsePositiveRate[-1]) * nn) / total)
        frames = pd.DataFrame({'true positive rate': truePositiveRate,
                               'false positive rate': falsePositiveRate,
                               'recall': recall,
                               'precision': precision,
                               'specificity': specificity,
                               'f1-measure': f1_measure,
                               'threshold': threshold,
                               'accuracy': accuracy})

        frames.to_csv('./csv/deploy_description.csv', sep=",", index=False)


def get_hdf5_data(dirpath):
    hdf5_files = os.listdir(dirpath)
    print(dirpath)
    arr_hdf5 = []
    for hdf5_file in hdf5_files:
        print(hdf5_file)
        arr_hdf5.append(list(HDF5Matrix(dirpath + '/' + hdf5_file, 'data')))
    return np.concatenate(arr_hdf5)


def autoencoder_mode():
    x_train = get_hdf5_data('./keras_hdf5/training')
    # x_train = get_hdf5_data('./keras_hdf5/validation')
    x_validation = get_hdf5_data('./keras_hdf5/validation')

    input_log = Input(shape=(114,))

    # compression (dense -> dropout -> leakyrelu)
    encoded = Dense(512, activation='linear')(input_log)
    encoded = Dropout(0.5)(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded)
    encoded = Dense(256, activation='linear')(encoded)
    encoded = Dropout(0.5)(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded)
    encoded = Dense(128, activation='linear')(encoded)
    encoded = Dropout(0.5)(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded)
    encoded = Dense(64, activation='linear')(encoded)
    encoded = Dropout(0.5)(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded)
    # encoded = Dense(32, activation='sigmoid')(encoded)
    # decoded = Dense(64, activation='linear')(encoded)
    # decoded = LeakyReLU(alpha=0.01)(decoded)

    # decompression (dense -> dropout -> leakyrelu)
    decoded = Dense(128, activation='linear')(encoded)
    decoded = Dropout(0.5)(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    decoded = Dense(256, activation='linear')(decoded)
    decoded = Dropout(0.5)(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    decoded = Dense(512, activation='linear')(decoded)
    decoded = Dropout(0.5)(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    decoded = Dense(114, activation='sigmoid')(decoded)

    autoencoder = Model(input_log, decoded)
    # encoder = Model(input_log, encoded)

    # encoded_input = Input(shape=(64,))
    # decoder_layer = autoencoder.layers[4]
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    print(autoencoder.summary())
    autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=6,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_validation, x_validation))

    autoencoder.save('autoencoder.h5')

    return

#autoencoder_mode()
deploy()
