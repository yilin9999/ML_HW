# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:15:41 2018

@author: yilin9999
"""
import numpy as np
import tensorflow as tf
import keras

from keras import optimizers
from keras import callbacks
from sklearn.model_selection import train_test_split

import keras
import time
import dv_analysis


"""
class DataGenerator(keras.utils.Sequence):
    def __init__(self, trainX, trainY, batchSize):        
        self.sampleCnt     = trainX.shape[0]        
        self.sampleOfBatch = np.floor(trainX.shape[0]/batchSize)
        self.batchSize     = batchSize
        self.counter       = 0
    
    def __len__(self):
        return self.sampleOfBatch
    
    def __getitem__(self):
        headIdx      = self.batchSize * self.counter
        tailIdx      = headIdx + self.batchSize
        self.counter = 0
        
        while 1:
            batchX = trainX[headIdx:tailIdx].toarray().astype(np.bool)
            batchY = np.array(trainY[headIdx:tailIdx])
            
            return batchX, batchY
"""     
    

