# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:44:44 2018

@author: yilin9999
"""
import numpy as np
import sys

import word_embeded

#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_train_data(filename):
    
    try:
        print("Loading %s" %filename)        
        tmpList = np.loadtxt(filename, dtype=np.str, delimiter=' +++$+++ ', encoding="utf-8")                
    except Exception as e: #BaseException
        print(type(e), str(e))
        sys.exit(-1)
    
    rawX  = tmpList[:,1]
    #rawY  = tmpList[:,0]
    
    trainX = word_embeded.bow(rawX)
    trainY = np.array(list(map(int,tmpList[:,0])))
    
    del rawX    
    del tmpList
   
    return trainX, trainY
    
    