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
    sampleCnt  = trainX.shape[0]
    featureCnt = trainX.shape[1]
    
    cnt0 = 0
    cnt1 = 0
    sum0   = np.zeros(featureCnt)
    sum1   = np.zeros(featureCnt)
    sigma0 = np.zeros((featureCnt, featureCnt))
    sigma1 = np.zeros((featureCnt, featureCnt))
    mu0 = np.zeros((featureCnt,))
    mu1 = np.zeros((featureCnt,))
        
    for i in range(sampleCnt):
        if trainY[i] == 0:
            cnt0 += 1
            sum0 += trainX[i] 
        else:
            cnt1 += 1
            sum1 += trainX[i] 
    
    mu0    = sum0/cnt0
    mu1    = sum1/cnt1        
    #mu0    = np.mean(trainX, axis=0) #avg of each column
    #mu1    = np.mean(trainX, axis=0) #avg of each column
    
    for i in range(sampleCnt):    
        if trainY[i] == 0:  
            sigma0 += np.dot(np.transpose([trainX[i]-mu0]), [(trainX[i]-mu0)])        
            #sigma0 += np.dot(np.transpose([trainX[i] - mu0]), [(trainX[i] - mu0)])
        else:
            sigma1 += np.dot(np.transpose([trainX[i] - mu1]), [(trainX[i] - mu1)])
            #sigma1 += np.dot(np.transpose([trainX[i]-mu1]), [(trainX[i]-mu1)])
            
    sigma0 /= cnt0
    sigma1 /= cnt1
    
    sh_sigma = (float(cnt0)/sampleCnt)*sigma0 + (float(cnt1)/sampleCnt)*sigma1    
        
    w, b, acc = gnerative_train(trainX, trainY, cnt0, cnt1, mu0, mu1, sh_sigma)

    print("Train acc = %.4f" %(acc))    
    
    testX = mydata.load_X("X_test.csv") 
    testY = mydata.load_testY("correct_answer.csv") 
    #mydata.normalization(testX)    
    mydata.evaluate(testX, testY, w, b)
    print("Test acc = %.4f" % acc)
    
    
def gnerative_train(trainX, trainY, cnt0, cnt1, mu0, mu1, sh_sigma):    
    sh_sigma_inv = np.linalg.inv(sh_sigma)
    
    w  = np.dot( (mu1-mu0), sh_sigma_inv)
 
    b = (-0.5) * np.dot( np.dot([mu1], sh_sigma_inv), mu1) + \
        0.5* np.dot( np.dot([mu0], sh_sigma_inv), mu0)  + \
        np.log(cnt1/cnt0)    
        
    acc = mydata.evaluate(trainX, trainY, w, b)
    return w, b, acc  


if __name__ == "__main__":
    main() 
    

