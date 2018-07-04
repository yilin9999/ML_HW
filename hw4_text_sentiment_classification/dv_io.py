# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:44:44 2018

@author: yilin9999
"""

import numpy as np
import pickle
import time
import sys

#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
#from scipy.sparse import hstack
#from scipy import sparse


from sklearn.model_selection import train_test_split


            
def dump_pkl(obj, filename):
    print("Saving %s: (%s)" %(type(filename), type(obj)))
    try:
        with open(filename, "wb") as fptr:
            pickle.dump(obj, fptr, pickle.HIGHEST_PROTOCOL)
            
    except Exception as e:
        print(type(e), str(e))  
        
        
def load_pkl(filename):
    
    obj = None
    try:
        with open(filename, "rb") as fptr:
            obj = pickle.load(fptr)                
    except Exception as e:
        print(type(e), str(e))
    
    print("Loading: %s: (%s)" %(type(filename), type(obj)))
    return obj


class DataManerger:
    def __init__(self):
        self.trainX = []
        self.trainY = []
        self.validX = []
        self.validY = []
        self.semiX =  []
        self.semiY =  []        
        self.testX =  []
                
        self.semiLabelY    = []        
        self.semiLabelYCnt = 0

    def append_new_data(self, filename, dataType):
        
        print("Loading %s" %filename)        
        
        try:
            with open(filename, 'r', encoding='utf-8') as fptr:
                    
                if dataType == 'train':
                    for line in fptr:
                        token = line.strip().split(' +++$+++ ')                    
                        #self.textData['train'].append()(np.bool(token[0]))
                        self.trainY.append(np.int8(token[0]))
                        self.trainX.append(token[1])
                elif dataType == 'semi':
                    #self.semiLearn = True
                    for line in fptr:
                        self.semiX.append(line.strip())
                        self.semiY.append(None)
                elif dataType == 'test':
                    next(fptr)
                    for line in fptr:                        
                        token = line.strip().split(',')
                        self.testX.append(line.strip())                
                else: 
                    print("[Error] Wrong dataType (only 'train', 'semi', test')")
                    sys.exit(-1)                           
                
        except Exception as e: #BaseException
            print(type(e), str(e))
            sys.exit(-1)    
            
            
    def split_valid_from_train(self, ratio, shuffle):                
        self.trainX, self.validX, self.trainY, self.validY = train_test_split(self.trainX, 
                                                                              self.trainY, 
                                                                              test_size=ratio,
                                                                              shuffle=shuffle)
        print("Train Data Size     :%d" %(len(self.trainX)))
        print("Validation Data Size:%d" %(len(self.validX)))
        
    def set_train_data(self, X, Y):
        self.trainX = X
        self.trainY = Y    
    
    def set_semi_dataX(self, X):
        self.semiX = X                  
    
    def set_semi_dataY(self, Y):        
        self.semiY = Y   
        self.semiLabelY.extend([False]*len(Y))
        #self.semiLabelCnt= len(Y)
        
    def get_train_data(self):
        return self.trainX, self.trainY

    def get_semi_dataX(self, X):
        self.semiX = X            
        
    def get_valid_data(self):
        return self.validX, self.validY
    
    
    def psudo_lable(self, predY, thd):
        print("Psudo-labling semi-supervised's data")
        print("Original Train size: %d, %d: " %(len(self.trainX), len(self.trainY)))
        print("Original Semi  size: %d, %d: " %(len(self.semiX),  len(self.semiY)))
        
        beginTime = time.time()
        #tmpCsr = sparse.csr_matrix(np.array([]))
        tmpX = []
        tmpY = []        
        semiNewX = []
        
        
        for i in range(len(predY)):            
            if self.semiY[i]==None and  (predY[i] >= thd or predY[i] <= (1-thd)):                                                 
                tmpX.append(self.semiX[i])
                
                label = np.int8(np.round(predY[i]))
                tmpY.append(label)
                self.semiY[i]=label
            else:
                semiNewX.append(self.semiX[i])           
            
            
        self.trainX.extend(tmpX)        
        self.trainY.extend(tmpY)       
        
        self.semiX = semiNewX
        self.semiY = [None] * len(semiNewX)
        
        #print(predY)               
        #print(tmpX)
        #print(tmpY)
        #print(self.semiX)
        
        endTime = time.time()
        print("New Train size: %d, %d: " %(len(self.trainX), len(self.trainY)))
        #print("New Semi  size: %d, %d: " %(len(self.semiX)-len(self.semiX)-self.semiLabelYCnt, len(self.semiY)-self.semiLabelYCnt))
        print("Psudo-labling Time: %.4f sec" %(endTime-beginTime))

    
            
"""
def load_train_data(filename, label=True):
    
    try:
        print("Loading %s" %filename)        
        tmpList = np.loadtxt(filename, dtype=np.str, delimiter=' +++$+++ ', encoding="utf-8")                
        
    except Exception as e: #BaseException
        print(type(e), str(e))
        sys.exit(-1)    
    
    
    #tmpList = tmpList[:30000]    
    #rawY  = tmpList[:,0]
    
    trainSprX = word_embeded.bow(tmpList[:,1])        
    trainY    = np.array(list(map(int,tmpList[:,0])))
         
    del tmpList

    print("Loading Finish")   
    return trainSprX, trainY    


class InputData():
    def __init__(self, dataType):        
        self.dataType = dataType
        self.dataX  = []        
        self.dataY  = []
        
    def append_new_data(self, filename):
        
        print("Loading %s" %filename)        
        
        try:
            with open(filename, 'r', encoding='utf-8') as fptr:
                    
                if self.dataType == 'train':
                    for line in fptr:
                        token = line.strip().split(' +++$+++ ')                    
                        #self.textData['train'].append()(np.bool(token[0]))
                        self.dataY.append(np.int8(token[0]))
                        self.dataX.append(token[1])
                elif self.dataType == 'semi':
                    #self.semiLearn = True
                    for line in fptr:
                        self.dataX.append(line.strip())
                        self.dataY.append(None)
                elif self.dataType == 'test':
                    next(fptr)
                    for line in fptr:                        
                        token = line.strip().split(',')
                        self.dataX.append(line.strip())                
                else: 
                    print("[Error] Wrong dataType (only 'train', 'semi', test')")
                    sys.exit(-1)                           
                
        except Exception as e: #BaseException
            print(type(e), str(e))
            sys.exit(-1)    
            
    
    def append_new_data_from_array(self, dataX, dataY, dataType):
        self.dataType = dataType
        self.dataX  = dataX        
        self.dataY  = dataY        
            
    def split_valid_from_train(self, ratio, shuffle):                
        self.dataX, validX, self.dataY, validY = train_test_split(self.dataX, 
                                                                  self.dataY, 
                                                                  test_size=ratio,
                                                                  shuffle=shuffle)
    
    
    
    def word_embeded(self, embededFuncPtr):                
        textDataAll = itertools.chain(self.trainTextX, self.semiTextX) 
        
        trainSprX = embededFuncPtr(textDataAll)      
        
        self.trainX     = trainSprX[0 : len(self.trainTextX)]            
        if self.semiLearn:
            self.semiX = trainSprX[len(self.trainTextX):]         
    

    #def chain_train_text_data(self):        
    #    return itertools.chain(self.trainTextX, self.semiTextX)    
   
    def get_dataXY(self):
        return self.dataX, self.dataY
    
    def get_dataX(self):
        return self.dataX
    
    #def get_dataY(self):
    #    return self.dataX
    
    #def get_valid_data(self):
    #    return self.validX, self.validY

    #def get_wordvec_dim(self):
    #    return self.trainX.shape[1]
    
    #def get_wordvec_num(self):
    #    if self.semiLearn==False:
    #        return self.trainX.shape[0]
    #    else:            
    #        return self.trainX.shape[0] + self.semiX.shape[0]
     
    #def split_valid_data(self, testSize, shuffle):
    #    self.trainX, self.validX, self.trainY, self.validY = train_test_split(self.trainX, self.trainY, test_size=testSize, shuffle=shuffle)
        
"""