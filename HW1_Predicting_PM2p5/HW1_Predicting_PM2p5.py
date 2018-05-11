# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
import time
from HW1_test import test_routing


testFile  = "./test.csv"
trainFile = "./train.csv"
ansFile   = "./ans.csv"


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
#weight_v  = np.zeros(featureCnt + 1) #include bias
weight_v = np.array([-9.80793890e-03, -1.29841030e-02, -7.38698906e-03, -8.81859736e-03,
       -2.38529500e-04, -9.87308515e-05,  1.64765496e-03,  1.70783440e-03,
        1.53192638e-02,  2.25598074e-02,  2.60717252e-02,  2.69105858e-02,
        2.40807860e-02,  3.20968169e-02,  3.90624427e-02,  3.75715200e-02,
        4.63208358e-02,  7.15388345e-02,  3.26108937e-02,  1.39473050e-02,
        4.92155379e-02, -8.85841261e-03,  9.48713881e-02, -3.25685815e-02,
       -3.36419335e-02,  2.40070545e-01,  5.40844027e-01, -4.77328143e-02,
       -4.42793951e-02, -4.22697461e-02,  1.61257714e-02,  9.67613361e-02,
        1.66473573e-03, -2.32409320e-02,  6.81090054e-02,  2.49591464e-01,
        5.31402530e-02,  2.58756326e-02,  8.40427795e-02,  3.03309255e-04,
       -5.78731853e-03, -2.67506426e-02,  4.56027044e-02,  2.72345125e-02,
       -7.96754882e-02, -4.85931396e-03, -2.48502943e-02, -2.26884732e-02,
       -2.40066761e-02,  2.67321736e-03, -1.61177876e-02, -5.71916976e-02,
       -5.79768741e-03,  1.51308317e-01,  2.07183032e-03, -1.53927960e-02,
        3.15775614e-04, -7.90437427e-03,  5.63767199e-03, -1.49247902e-02,
       -3.63922995e-02,  9.45629584e-03,  1.12917358e-01,  8.56968889e-03,
        4.34501470e-03, -7.14275434e-03, -1.13035666e-02, -1.24031086e-02,
       -2.91948198e-02, -2.94777230e-02,  1.25141025e-02,  9.45388272e-02,
       -9.44139759e-03,  1.02562327e-02,  7.86727538e-03, -3.49865261e-03,
       -1.25423193e-02,  1.57573061e-02, -2.37005405e-03, -1.74748769e-02,
        6.90887692e-02,  3.03713185e-03,  2.12670869e-02,  2.61550228e-02,
       -4.40107867e-02,  4.58582071e-02,  1.75462199e-01, -3.23595300e-01,
        9.09746053e-02,  8.14050901e-01,  2.51707853e-02,  3.10340119e-03,
       -5.84329895e-02, -9.18648547e-03, -3.68417047e-02,  5.14117549e-02,
        3.32213194e-03, -3.12360409e-02, -8.90619293e-02,  5.26264664e-03,
        3.78956918e-03, -4.71702582e-03, -9.93375338e-03, -9.70803282e-03,
       -3.86446383e-03, -8.80585854e-03,  3.37922562e-03,  8.93087849e-03,
       -7.23624782e-02,  7.91173479e-02, -1.38846056e-02, -3.31961005e-02,
       -4.32227064e-05,  1.62622020e-02, -4.88924656e-02,  6.76185804e-02,
        2.18017162e-01,  2.06084960e-02,  1.67944301e-02,  1.79664539e-02,
        1.75600179e-02,  3.89943485e-02,  3.97036649e-02,  3.17073756e-02,
        4.31226156e-02,  8.13359702e-02, -5.55614675e-04,  2.98002720e-03,
       -9.37060179e-04,  2.33032069e-03,  3.80310726e-04,  1.70820400e-03,
       -2.52870782e-03,  1.10890284e-03,  5.61178848e-04, -1.73360296e-03,
       -6.68486441e-04,  6.17412803e-04, -2.52518449e-03,  1.02024313e-03,
        3.09191179e-04, -5.47505189e-04, -2.89858136e-03,  5.12806550e-04,
       -2.70286650e-02, -2.93798533e-03,  1.61928019e-02, -4.24494128e-03,
       -2.70109661e-03, -1.35305664e-02, -4.61995114e-02, -5.06953778e-02,
       -3.02702888e-02, -3.82138784e-02, -6.07748862e-04, -4.26517961e-02,
       -3.41259438e-02,  6.29304634e-03,  4.99911890e-02, -3.38923357e-02,
       -5.98978578e-02, -5.99047311e-03,  2.60541805e-02])
    
trainX_T  = np.transpose(trainX)

######## Learning Perameters
epoch       = 10000
Stoch       = 1
l_rate      = float(1)


trainXp    = trainX
trainYp    = trainY
trainXp_T  = trainX_T

if Stoch == 1:    
    s_gra     = 1
    batchsize = 1
    showPeried  = 30   
else:    
    s_gra       = featureCnt + 1
    batchsize   = trianCnt    
    showPeried  = 1000

checkPeriod = showPeried * 5 
iteration = trianCnt//batchsize
#epochN    = 1000*iteration

######## Start Training

trainTimeB = time.time()
for i in range(epoch):    
#for i in range(20):    
    
    for j in range(iteration):
    #### Stochastic Gradient desent
        if Stoch == 1: 
            txIdx      = i%trianCnt
            trainXp    = trainX[txIdx]
            trainYp    = trainY[txIdx]
            trainXp_T  = trainXp   #1d array transpose (cannot use np.transpose)
        
        #### Batch Gradient desenct    
        hypo_v   = np.dot(trainXp, weight_v)
        diff_v   = hypo_v - trainYp
        #print(trainX_T)
        #print(weight_v)
        gra      =  2* np.dot(trainXp_T, diff_v)    
        
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        
        l_rate *= 1/np.sqrt(1.0e-8*i+1)    
        weight_v = weight_v - l_rate * (gra/ada)    
        #break
        #print(l_rate)
    
    #### Monitor
    if(i % showPeried == 0):
        if Stoch == 1:        
            hypo_v   = np.dot(trainX, weight_v)
            diff_v   = hypo_v - trainY       
        Loss     = np.sqrt(np.mean(diff_v**2))                
        print("%d interation: l_rate=%.4f, Loss = %.4f" %(i, l_rate, Loss))
        
        if (i % checkPeriod == 0):
            test_routing(weight_v, testFile, ansFile, Loss)
        
#end for i in range(1):         
trainTimeE = time.time()

test_routing(weight_v, testFile, ansFile, Loss)


print("Training Time: %.4f sec" %(trainTimeE - trainTimeB))







"""

u = Lr * (y_data[n] - (w*x_data[n] + b)) << 1  # 2u
w_grad = w_grad - u * x_data[n]
b_grad = b_grad - u

def computeCost(x, y, theta):
"""
