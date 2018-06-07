# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 09:13:20 2018

@author: yilin9999
"""
from keras.models import Sequential
from keras.layers import Dense

def create_dnn(paraDict):
    
    print("Create DNN model")
        
    model = Sequential()
    
    model.add(Dense(input_dim=paraDict['vecSize'], units=paraDict['neuroCnt'], activation='relu'))
    
    for i in range(2):
        model.add(Dense(units=1024, activation='relu'))
    
    model.add(Dense(output_dim=1, activation='sigmoid'))
    
    model.summary()
    
    return model
        
    
    
    
    
    