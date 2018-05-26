# -*- coding: utf-8 -*-
"""
Created on Fri May 25 00:06:25 2018

@author: yilin9999
"""

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils 

class MyData:
    def __init__(self, filename, normalize=1):            
        self.seed   = 0
        self.load_data(filename, normalize=1)
        
    def load_data(self, filename, normalize=1):
        #input: filename = str
        
        trainAll = np.loadtxt(filename, skiprows=1, dtype=np.str, delimiter=",")        
        #trainAll = trainAll[:100]
        
        tempList = []
        for itr in trainAll[:,1]:
            tempList.append(itr.split(' '))

        if normalize==1:            
            self.trainX = np.array(tempList).astype(np.float32)                                                 
            self.trainX /= 256            
        else:            
            self.trainX = np.array(tempList).astype(np.uint8)                                   
            
            
        self.trainX = self.trainX.reshape(self.trainX.shape[0], 48, 48, 1)
        self.trainY = trainAll[:,0].astype(np.uint8)        
        
        #one-hot coding for output
        self.trainY = np_utils.to_categorical(self.trainY)
        
        del trainAll
        #return self.trainX, self.trainY
    
    def split_valid_data(self, valid_ratio):
        
        if valid_ratio > 1 or valid_ratio < 0:            
            raise ValueError("valid_ratio must be between 0 to 1")
        
        """
        tX, vX, tY, vY =  train_test_split(self.trainX, self.trainY, 
                                           test_size=valid_ratio, 
                                           shuffle=False,
                                           random_state=self.seed)    
        """
        th = np.int(np.floor(self.trainX.shape[0] * (1-valid_ratio)))
        
        tX = self.trainX[:th]
        vX = self.trainX[th:]
        tY = self.trainY[:th]
        vY = self.trainY[th:]
        
        return tX, vX, tY, vY            
        
        
    def get_train_data(self):
        return self.trainX, self.trainY
    
    def get_valid_data(self):
        return self.trainX, self.trainY    
    
        


    
    
    
    
    
    
    
    
    
