# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
from itertools import chain



#Read file
data_str   = np.loadtxt("train.csv", skiprows=1, dtype=np.str, delimiter=",")
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

train_x = np.zeros(shape=(trainCnt, featureCnt))
train_y = np.zeros(trainCnt)

c     = 0
start = 0

for i in range(12):
    start = (i*12) * (24*20)
    for j in range(mUnit):
        for k in range(18):                                        
            ks = 9 * k            
            train_x[c][ks:ks+9] = data_float[k][start:start+9]
            train_y[c]          = data_float[9][start+10]
            #print ([i_month, j_mUnit, k, ks, start, start+9])
        c += 1
        start += 1
        
np.savetxt("tmp.csv", train_x, fmt ='%.2f', delimiter=",")

weight     = []
bias       = []
iteration  = 1000

#for i in range(iteration):


#print(data_str[1])


"""
Lr = 1
u = Lr * (y_data[n] - (w*x_data[n] + b)) << 1  # 2u
w_grad = w_grad - u * x_data[n]
b_grad = b_grad - u

def computeCost(x, y, theta):
"""
