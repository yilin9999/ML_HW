# -*- coding: utf-8 -*-
"""
Created on Fri May 18 10:35:02 2018

@author: yilin9999
"""

import mydata
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import callbacks

from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

def main():
    #end def sigmoid():
    trainX = mydata.load_X("X_train.csv")
    trainY = mydata.load_Y("Y_train.csv")
    mydata.normalization(trainX)  
    #trainX = trainX[5:9]
    #trainY = trainY[5:9]
    
    epochNum    = 100
    seed        = 0
    valid_ratio = 0.1
    dropRate    = 0.5
    sampleCnt  = trainX.shape[0]
    featureCnt = trainX.shape[1]
    
    #include shuffling
    trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=valid_ratio, random_state=seed)
        
    myModel = Sequential()
    myModel.add(Dense(input_dim=featureCnt, units=500, activation='relu'))    
    #layers.Dropout(dropRate, noise_shape=None, seed=None)    
    for i in range(1):
        myModel.add(Dense(units=featureCnt,
                          kernel_regularizer=regularizers.l2(0.1),
                          activation='relu'))    
        #myModel.add(Dropout(0.5))        
    
    myModel.add(Dense(1, activation='sigmoid'))
    
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-6, amsgrad=False)
    #adag = optimizers.Adagrad(lr=0.01)    
    #history = LossHistory()
    myModel.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    
    myEarlyStop = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto')   
    
    myHistory = myModel.fit(trainX, trainY, epochs=epochNum, validation_data=(validX, validY), 
                            batch_size=100, shuffle=True, verbose=2, callbacks=[myEarlyStop])        
    
    mydata.plot_result(myHistory)


    testX = mydata.load_X("X_test.csv") 
    testY = mydata.load_testY("correct_answer.csv") 
    mydata.normalization(testX)    
    loss, acc = myModel.evaluate(testX, testY, verbose=1)
    print("Final acc = %.4f" % acc)
    

if __name__ == "__main__":
    main() 
