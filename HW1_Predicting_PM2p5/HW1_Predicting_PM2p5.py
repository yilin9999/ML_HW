# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
import time
from HW1_test import test_routing


testFile  = "test.csv"
trainFile = "train.csv"
ansFile   = "ans.csv"


#Read Training file
data_str   = np.loadtxt(trainFile, skiprows=1, dtype=np.str, delimiter=",")
data_float = [[] for x in range(18)]

row = 0
for it_row in data_str:        
    for it_str in it_row[3:]:        
        if it_str == "NR":
            data_float[row%18].append(0)
        else:
            data_float[row%18].append(float(it_str))   
    row += 1

# Check data 
col_size = len(data_float[0])
 
for idx, item in enumerate(data_float[1:], start=1):
    if col_size != len(item):
        print("%d: %s has different size %d" % idx, item, len(item));


#Transform x_train
mUnit      = 20*24 - 9
trainCnt   = 12 * mUnit
featureCnt = 18*9

trainX = np.zeros(shape=(trainCnt, featureCnt))
trainY = np.zeros(trainCnt)

c     = 0
start = 0


for i in range(12):
    start = i * (24*20)
    for j in range(mUnit):
        for k in range(18):                                        
            ks = 9 * k            
            trainX[c][ks:ks+9] = data_float[k][start:start+9]
            trainY[c]          = data_float[9][start+9]
            #print ([i_month, j_mUnit, k, ks, start, start+9])
        c += 1
        start += 1


#include bias
trainX   = np.concatenate( \
                (trainX, np.ones((len(trainX), 1), dtype=float)),  \
                axis=1)

#Modify size
#trianCnt = 10
trianCnt = len(trainX)
trainX   = trainX[:trianCnt]
trainY   = trainY[:trianCnt]

#np.savetxt("tmp_x.csv", trainX, fmt ='%.2f', delimiter=",")        
#np.savetxt("tmp_y.csv", trainY, fmt ='%.2f', delimiter=",")        

#initialize Training parameters
weight_v  = np.zeros(featureCnt + 1) #include bias
iteration   = 100000
l_rate      = 1
#s_gra       = featureCnt + 1
s_gra       = featureCnt + 1
trainX_T    = np.transpose(trainX)
Stoch       = 0

#Start Training
trainXp    = trainX
trainYp    = trainY
trainX_Tp  = trainX_T
trainTimeB = time.time()

for i in range(iteration):    
    
    #### Stochastic Gradient desent
    if Stoch == 1: 
        txIdx      = i%trianCnt
        trainXp    = trainX[txIdx]
        trainYp    = trainY[txIdx]
        trainX_Tp  = np.transpose(trainXp)
    
    #### Batch Gradient desenct    
    hypo_v   = np.dot(trainXp, weight_v)
    diff_v   = hypo_v - trainYp
    #print(trainX_T)
    #print(weight_v)
    gra      =  2* np.dot(trainX_Tp, diff_v)    
    
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    weight_v = weight_v - l_rate * gra/ada
    
    
    if(i% 1000==0):
        Loss     = np.sqrt(np.sum(diff_v**2)/trianCnt)
        print("%d interation: Loss = %.4f" %(i, Loss))
        
trainTimeE = time.time()
print("Training Time: %.4f sec" %(trainTimeE - trainTimeB))


test_routing(weight_v, testFile, ansFile)




"""

u = Lr * (y_data[n] - (w*x_data[n] + b)) << 1  # 2u
w_grad = w_grad - u * x_data[n]
b_grad = b_grad - u

def computeCost(x, y, theta):
"""
