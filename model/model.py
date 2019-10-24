#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, LeakyReLU, Dropout
from keras.models import Model, load_model
from keras.utils.io_utils import HDF5Matrix

from sklearn import metrics

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import math


def l1_distance(u, v):
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
        if y[idx] == 1:
            b = -math.log10(p[idx])
            result += b
        elif p[idx] == 0:
            print(p[idx], p[idx] == 0)
            result += 0
        else:
            print(p[idx], y[idx])
            b = -(y[idx]*math.log10(p[idx]) + (1-y[idx])*math.log10(1-p[idx]))
            result += b
    return result


def deploy(model_dir, makeCSV=True, makePlot=True) :
    model = load_model(model_dir)
    normalDataLoss = []
    attackDataLoss = []

    hdf5_files = os.listdir('./hdf5/test')
    total = 0
    nn = 0
    an = 0

    for hdf5_file in hdf5_files:
        print(hdf5_file)
        data_test = list(HDF5Matrix('./hdf5/test/'+hdf5_file, 'data'))
        label_test = list(HDF5Matrix('./hdf5/test/'+hdf5_file, 'label'))
        total = total+len(data_test)
        for idx in range(len(data_test)):
            input_test = data_test[idx].reshape((1, 114))
            predict_test = model.predict(input_test)
            loss = l1_distance(input_test, predict_test)
            if label_test[idx] == 0:  # normal 0
                normalDataLoss.append(loss)
                nn = nn+1
            else:
                attackDataLoss.append(loss)
                an = an+1
    print("Normal data loss(%d): %f" % (len(normalDataLoss), np.average(np.array(normalDataLoss))))
    print("Attack data loss(%d): %f" % (len(attackDataLoss), np.average(np.array(attackDataLoss))))
    print("total data: %d %d %d" % (total, nn, an))
    allDataLoss = normalDataLoss+attackDataLoss

    print("Sum : ", len(allDataLoss))
    allLabel = [0]*len(normalDataLoss)+[1]*len(attackDataLoss)

    fpr, tpr, thresholds = metrics.roc_curve(np.array(allLabel), np.array(allDataLoss), drop_intermediate=False)
    #print(metrics.confusion_matrix())

    auc = metrics.auc(fpr, tpr)
    print("auc", auc)

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
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('./plot/' + model_dir + '.png', dpi=80)
        # plt.show()

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
        for rate in range(10, 21, 1) :
            truePositiveRate.append(tpr[np.where(tpr >= (rate*0.05))[0][0]])
            falsePositiveRate.append(fpr[np.where(tpr >= (rate*0.05))[0][0]])
            recall.append(truePositiveRate[-1])
            precision.append((truePositiveRate[-1]*len(attackDataLoss))/(truePositiveRate[-1]*len(attackDataLoss) +
                                                                         falsePositiveRate[-1]*len(normalDataLoss)))
            specificity.append(1-falsePositiveRate[-1])
            f1_measure.append((2*recall[-1]*precision[-1])/(precision[-1]+recall[-1]))
            threshold.append(thresholds[np.where(tpr >= (rate*0.05))[0][0]])
            accuracy.append((truePositiveRate[-1] * an + (1-falsePositiveRate[-1]) * nn)/total)
        frames = pd.DataFrame({'true positive rate' : truePositiveRate,
                      'false positive rate' : falsePositiveRate,
                      'recall' : recall,
                      'precision' : precision,
                      'specificity' : specificity,
                      'f1-measure' : f1_measure,
                      'threshold' : threshold,
                      'accuracy' : accuracy})

        frames.to_csv('./csv/' + model_dir + '.csv', sep="\t", index=False)


def get_hdf5_data(dirpath):
    hdf5_files = os.listdir(dirpath)
    print(dirpath)
    arr_hdf5 = []
    for hdf5_file in hdf5_files:
        print(hdf5_file)
        arr_hdf5.append(list(HDF5Matrix(dirpath+'/'+hdf5_file, 'data')))
    return np.concatenate(arr_hdf5)

def autoencoder_mode(loss, train_type):
    if train_type == 'normal':
          x_train = get_hdf5_data('./hdf5/train')
    else:
        x_train = get_hdf5_data('./hdf5/train_with_attack')
    x_validation = get_hdf5_data('./hdf5/test')

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

    print(autoencoder.summary())
    autoencoder.compile(optimizer='rmsprop', loss=loss) # adam이랑도 비교해보자.

    autoencoder.fit(x_train, x_train,
                    epochs=6,
                    batch_size=64,
                    validation_data=(x_validation, x_validation))

    autoencoder.save('autoencoder_' + loss + '_' + train_type + '.h5')
    return
