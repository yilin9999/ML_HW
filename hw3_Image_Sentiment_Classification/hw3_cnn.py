# -*- coding: utf-8 -*-
"""
Created on Fri May 25 00:06:25 2018

@author: yilin9999
"""

import numpy as np
from hw3_data import MyData

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras import optimizers
from keras import callbacks
    
def create_cnn(featureCnt, ClassCnt):
    imgShape = (48,48,1) #((batch, height, width, channels))
    kernelSize=(3,3)
    
    
    cnnModel = Sequential()
    
    cnnModel.add(Conv2D(filters=16, 
                        kernel_size=kernelSize, 
                        input_shape=imgShape,    #default is channel last
                        padding="same",                                        
                        activation='relu'                   
                        ))
    
    cnnModel.add(MaxPooling2D(pool_size=(2, 2), strides=1))    
    
    cnnModel.add(Conv2D(filters=32, 
                        kernel_size=kernelSize, 
                        padding="same",
                        activation='relu', 
                        ))    
    
    cnnModel.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    
    cnnModel.add(Dropout(0.2))
    cnnModel.add(Flatten())        
    cnnModel.add(Dense(512, activation='relu'))    
    cnnModel.add(Dense(128, activation='relu'))
    cnnModel.add(Dense(output_dim=ClassCnt, activation='softmax'))

    cnnModel.summary()  
    
    return cnnModel


def main():
    
    #trainData = MyData("C:\\testdata\\train_small.csv")    
    trainData = MyData("C:\\testdata\\train.csv")    
    
    trainX, validX, trainY, validY = trainData.split_valid_data(0.2)
    
    featureCnt = trainX.shape[1]
    ClassCnt   = 7
    epochNum   = 10    
    
    cnnModel = create_cnn(featureCnt, ClassCnt)
    
    myEarlyStop = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    adam = optimizers.adam(lr=0.001)
    cnnModel.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = cnnModel.fit(trainX, trainY, 
                           epochs=epochNum, 
                           batch_size=10,
                           validation_data=(validX, validY),
                           callbacks=[myEarlyStop],
                           shuffle=True,
                           verbose=2)
    
    
if __name__ == "__main__":
    main()