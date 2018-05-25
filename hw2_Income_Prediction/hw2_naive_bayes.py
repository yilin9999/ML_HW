# -*- coding: utf-8 -*-
"""
Created on Sun May 13 20:19:03 2018

@author: yilin9999
"""

import mydata
import dModel
import numpy as np

def main():
    #end def sigmoid():
    trainX = mydata.load_X("X_train.csv")
    trainY = mydata.load_Y("Y_train.csv")
    mydata.normalization(trainX)
    #trainX = trainX[5:9]
    #trainY = trainY[5:9]        
   
    

if __name__ == "__main__":
    main() 
    

