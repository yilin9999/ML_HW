# -*- coding: utf-8 -*-
"""
Created on Fri May 25 00:06:25 2018

@author: yilin9999
"""

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils 
import matplotlib.pyplot as plt
from os import path
import pickle

class MyData:
    def __init__(self, filename, normalize=1):            
        self.seed   = 0
        self.load_data(filename, normalize=1)
        
    def load_data(self, filename, normalize=1):
        #input: filename = str
        
        try:
            trainAll = np.loadtxt(filename, skiprows=1, dtype=np.str, delimiter=",")        
            print("Loading %s" %filename)
        except:
            print("Cannot open %s" %filename)
            exit(0)
        #trainAll = trainAll[:100]
        
        tempList = []
        for itr in trainAll[:,1]:
            tempList.append(itr.split(' '))

        if normalize==1:            
            self.trainX = np.array(tempList).astype(np.float32)                                                 
            self.trainX /= 256            
        else:            
            self.trainX = np.array(tempList).astype(np.uint8)                                   
            
        #reshape   
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
    
    def plot_result(self, readDict):
        plt.plot(readDict['acc'])
        plt.plot(readDict['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(readDict['loss'])
        plt.plot(readDict['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    def write_result(self, myHistory, paraDict, runtime):
        #filename_train = 'loss_history_train.txt'
        #filename_test  = 'loss_history_loss.txt'
        
        with open("history.pkl","wb") as fptr:             
            pickle.dump(myHistory.history, fptr)                    
            
        if path.isfile("history.log"):            
            newfile = 0
        else:
            print("Create history.log")
            newfile = 1
            
        paraDict['Time']      = "%d"   % np.round(runtime)
        paraDict['Train_acc'] = "%.4f" % myHistory.history['acc'][-1]
        paraDict['Val_acc']   = "%.4f" % myHistory.history['val_acc'][-1]
        
        with open("history.log","a") as fptr:
            #printList = []
            
            if newfile==1:
                for itr in paraDict.keys():                                        
                    fptr.write("%10s" % itr)                    
                
                for i in range(paraDict['epochNum'])                    :
                    outstr = '#%d' % i
                    fptr.write("%10s" %outstr)
                fptr.write("\n")
           
            for itr in paraDict.keys():                 
                fptr.write("%10s" % str(paraDict[itr]))            
                 
            for itr in myHistory.history['acc']:                                
                fptr.write("%10.4f" % itr)

            fptr.write("\n")
            #printList.extend(myHistory.history['loss'])            
            #fptr.writelines(printList)
        
        #title_list = []
            

    def read_result(self):
        with open("history.pkl","rb") as fptr:
            readDict = pickle.load(fptr)
        """
        try: 
            myHistory = pickle.load(fptr)
        except (EnvironmentError, pickle.PicklingError) as err:            
            print("Cannot open %s" %filename)
        finally:
            fptr.close()
        """
        self.plot_result(readDict)
        
             


    
    
    
    
    
    
    
    
    
