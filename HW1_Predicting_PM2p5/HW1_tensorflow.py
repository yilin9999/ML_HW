# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:21:52 2018

@author: yilin9999
"""
import csv 
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


testFile  = "./test.csv"
trainFile = "./train.csv"
ansFile   = "./ans.csv"


#Read Training file
data_str   = np.loadtxt(trainFile, skiprows=1, dtype=np.str, delimiter=",")
data_float = [[] for x in range(18)]

row = 0
for it_row in data_str:        
    for it_str in it_row[3:]:        
        if it_str == "NR":
            data_float[row%18].append(0)
        else:
            data_float[row%18].append(float(it_str))   
    row += 1

# Check data 
col_size = len(data_float[0])
 
for idx, item in enumerate(data_float[1:], start=1):
    if col_size != len(item):
        print("%d: %s has different size %d" % idx, item, len(item));


#Transform x_train
mUnit      = 20*24 - 9
trainCnt   = 12 * mUnit
featureCnt = 18*9

trainX = np.zeros(shape=(trainCnt, featureCnt))
trainY = np.zeros(trainCnt)

c     = 0
start = 0

for i in range(12):
    start = i * (24*20)
    for j in range(mUnit):
        for k in range(18):                                        
            ks = 9 * k            
            trainX[c][ks:ks+9] = data_float[k][start:start+9]
            trainY[c]          = data_float[9][start+9]
            #print ([i_month, j_mUnit, k, ks, start, start+9])
        c += 1
        start += 1

trainX = np.concatenate(
        (trainX, np.ones((len(trainX), 1), dtype=float)), 
        axis=1)


epochNum = 100000

myModel = Sequential()
myModel.add(Dense(output_dim=1, input_dim=163))

#sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
myModel.compile(loss='mse', optimizer=adam)
#myModel.fit(trainX, trainY, nb_epoch=epochNum, batch_size=5620, verbose=0)


#Training
for i in range(100000):
    Loss = myModel.train_on_batch(trainX, trainY)
    if i%1000 == 0:
        print("%d interation: Loss=%.4f" %(i, Loss))    