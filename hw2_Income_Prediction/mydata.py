# -*- coding: utf-8 -*-
"""
Created on Sun May 13 20:41:24 2018

@author: yilin9999
"""
import numpy as np
import matplotlib.pyplot as plt
import random

def load_X(filename):
    return np.loadtxt(filename, skiprows=1, dtype=float, delimiter=",")
    
def load_Y(filename):
    return np.loadtxt(filename, skiprows=1, dtype=bool, delimiter=",")

def load_testY(filename):
    yt = np.loadtxt(filename, skiprows=1, dtype=bool, delimiter=",")
    return yt[:,1]
    

def shuffle_data(X, Y):
    np.random.seed(0)
    randIdx = np.arange(len(X))
    np.random.shuffle(randIdx)
    
    return (X[randIdx], Y[randIdx])
    
    
def plot_result(myHistory):
    plt.plot(myHistory.history['acc'])
    plt.plot(myHistory.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(myHistory.history['loss'])
    plt.plot(myHistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def split_valid_set(trainX_all, trainY_all, valid_ratio=0.1):
    sampleCnt  = trainX_all.shape[0]
    featureCnt = trainX_all.shape[1]
    
    th = int(np.floor(sampleCnt * (1-valid_ratio)))
    
    trainX_all, trainY_all = shuffle_data(trainX_all, trainY_all)
    trainX = np.array(trainX_all[:th])
    validX = np.array(trainX_all[th:])
    
    trainY = np.array(trainY_all[:th])
    validY = np.array(trainY_all[th:])
    
    return trainX, trainY, validX, validY
    
         

def sigmoid(z):
    #itype a: float    
    res = 1/((1.0) + np.exp(-z))
    return np.clip(res, 1e-10, 1e10) #avoid precision overflow

        
def normalization(X):
    mu    = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    sigma[sigma==0] = 1 # replace sigma 0
    X[:] = (X- [mu]) / [sigma] # using X[:] to avoid call by assignment (use call by reference)
    
    

def validation(X, Y, w, b):
    acc = evaluate(X, Y, w, b)
    print("acc=%.4f" %(acc))      
    
    
def evaluate(X, Y, w, b):    
    #np.mean((trainY-y)**2)    
    z  = np.dot(X, w) + b   
    y_ = np.around(sigmoid(z))
    
    result = np.zeros((2, Y.shape[0]))
    result[0] = (np.squeeze(Y) == y_)
    result[1] = Y
        
    
    #print("save result.csv")
    #np.savetxt('result.csv', result , fmt='%d' , delimiter = ',')    
    
    acc = result[0].sum()/Y.shape[0]    
    return  acc
    
    