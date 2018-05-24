# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:48:00 2018

@author: yilin9999
"""

import mydata
import dModel
import numpy as np
import time

def main():
    print("HIHI")
    trainX_all = mydata.load_X("X_train.csv")
    trainY_all = mydata.load_Y("Y_train.csv") 
    
    trainX, trainY, validX, validY = mydata.split_valid_set(trainX_all, trainY_all, valid_ratio=0.1)
    
    mydata.normalization(trainX)
    mydata.normalization(validX)
    #w, b = logistic_train(trainX, trainY, epoch=10000, lr = 0.01, batchSize=trainX.shape[0])
    
    begin = time.time()
    w, b = logistic_train(trainX, trainY, epoch=5000, lr = 1, regu=[2,1], batchSize=len(trainX))    
    end = time.time()
    print("Training time: %f sec" % (end-begin))
    
    print("validation result:")
    mydata.validation(validX, validY, w, b)  #bias is in the trainX    

    
    testX = mydata.load_X("X_test.csv") 
    testY = mydata.load_testY("correct_answer.csv") 
    mydata.normalization(testX)        
    
    #mydata.validation(trainX, trainY, w, b)  #bias is in the trainX    
    
    
    
    print("test result:")
    mydata.validation(testX, testY, w, b)  #bias is in the trainX    
      
    

def logistic_train(trainX, trainY, epoch=1, lr = 0.001, regu=[0, 0], batchSize=1):
    
    s_gra    = 0
    batchNum = trainX.shape[0] // batchSize
    featureCnt = trainX.shape[1]
    trainX_    = np.concatenate( \
                                (trainX, np.ones((len(trainX), 1), dtype=float)),  \
                                axis=1)            
    
    w = np.zeros(featureCnt + 1) #including bias    
    
    for i in range(epoch):                
        for j in range(batchNum):
            hptr = j*batchSize
            trainX_batch = trainX_[hptr:hptr+batchSize]
            trainY_batch = trainY[hptr:hptr+batchSize]
            
            z     = np.dot(trainX_batch, w.T)
            y     = mydata.sigmoid(z)
            diff  = trainY_batch - y
            
            regular = regu[0] * regu[1] * w 
            gra     = -1 * np.dot(trainX_batch.T, diff) + regular            
            #gra2  = -1 * np.dot(trainX_batch.T, diff.reshape(batchSize,1))
            
            s_gra += (gra**2)            
            ada   = np.sqrt(s_gra)         
            w = w  - lr * (gra/ada)            

        
        #calculate cross entropy
        if i%20==0:
            z     = np.dot(trainX_, w.T)
            y     = mydata.sigmoid(z)
            
            #loss = -1 * (np.dot(trainY, np.log(y)) + np.dot((1-trainY),np.log(1-y)))
            loss = -1 * (np.dot(np.squeeze(trainY), np.log(y)) + np.dot((1 - np.squeeze(trainY)), np.log(1 - y)))
            
            acc = mydata.evaluate(trainX, trainY, w[:-1], w[-1])  #bias is in the trainX
            print("Epoch %d, Loss=%.4f, acc=%.4f" %(i,loss, acc))
    
    return w[:-1], w[-1] #separate w and b


if __name__ == "__main__":
    main() 
 
#end of logistic_train

    
