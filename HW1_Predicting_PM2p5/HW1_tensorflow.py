# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:21:52 2018

@author: yilin9999
"""
import csv 
import numpy as np
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from HW1_test import test_routing
from keras.utils import plot_model
import sys
from time import time

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

############# Read TrainXY
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

#trainX = np.concatenate( \
#        (trainX, np.ones((len(trainX), 1), dtype=float)),  \
#        axis=1)

############# Read TestX
#Read Testing file
fptr  = open(testFile, "r")
frow  = csv.reader(fptr, delimiter=",")


tmpList = []
rowCnt  = 0
for irow in frow:    
    if rowCnt%18==0:
        tmpList.append([])
        
    for item in irow[2:11]:
        #print(item)
        if item == "NR":
            tmpList[-1].append(float(0))
        else:
            tmpList[-1].append(float(item))

    rowCnt += 1

fptr.close()

testX = np.array(tmpList)
#testX = np.concatenate((testX, np.ones((len(testX),1))), axis=1)
    
############# Read TestY
tmpList = []
fptr  = open(ansFile, "r")
frow  = csv.reader(fptr, delimiter=",")  

fptr.readline() #skip the first line
for irow in frow:     
    tmpList.append(float(irow[1]))

testY = np.array(tmpList)
fptr.close()
    
 ########################################################################################### 


#myWeight  = np.zeros(163) #include bias

myModel = Sequential()
myModel.add(Dense(1, input_dim=162, activation='linear'))

#myModel.layers[0].set_weights(myWeight)
#sys.exit(0)
#myModel.add(Dense(input_dim=163, output_dim=1))


#sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.5, nesterov=True)
class LossHistory(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            # you can access loss, accuracy in self.params['metrics']
            print('\n{}/{} - loss ....\n'.format(self.seen, self.params['nb_sample'])) 

epochNum = 100000
sgd = optimizers.SGD(lr=0.0001, decay=0.000, momentum=0.5, nesterov=None)
adag = optimizers.Adagrad(lr=0.01)
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)

#myModel.compile(loss='mse', optimizer=adam)
#myModel.fit(trainX, trainY, nb_epoch=epochNum, batch_size=5620, verbose=1)
#Training

history = LossHistory()
myModel.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

beginTime = time()
#myModel.fit(trainX, trainY, nb_epoch=epochNum, batch_size=200, shuffle=True, verbose=0)
for i in range(epochNum):
    Loss = myModel.train_on_batch(trainX, trainY)[0]
    if i%500 == 0:
        print("%d interation: Loss=%.4f" %(i, np.sqrt(Loss)))    
       
endTime = time()

Loss = myModel.evaluate(trainX, trainY, batch_size=5620)[0]

RMSE = np.sqrt(Loss)
print("Loss=%.4f" %(RMSE))    
print("Training Time: %.4f sec" %(endTime - beginTime))

score  = myModel.evaluate(testX, testY, batch_size=240)[0]
w, b = myModel.layers[0].get_weights()
print("Eout=%.4f" %(np.sqrt(score)))   

ww = np.ndarray.flatten(w) 
ww = np.concatenate((ww,b))
test_routing(ww, testFile, ansFile, RMSE)
 

