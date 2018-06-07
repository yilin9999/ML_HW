# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:22:54 2018

@author: yilin9999
"""
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


def bow(rawX):
    wordTok = CountVectorizer()    
    
    #print(rawX[0:5])
    bowArray = wordTok.fit_transform(rawX)
    #print("==============")
    #print(wordTok.get_feature_names())
    #print(X.toarray().astype(np.bool))
    return bowArray.toarray().astype(np.bool)
    