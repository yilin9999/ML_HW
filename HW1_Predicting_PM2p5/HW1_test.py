# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:21:52 2018

@author: yilin9999
"""
import csv 
import numpy as np
#

    
def test_routing(myModel, testFile, ansFile, trainLoss):
    #:type myModel:  np.array 163 x 1
    #:type testFile: str
    #:type trainLoss: float
    #:type ansFile: str    

    #Read Testing file
    fptr  = open(testFile, "r")
    frow  = csv.reader(fptr, delimiter=",")
    
    ############# Read TestX
    tmpList = []
    rowCnt  = 0
    for irow in frow:    
        if rowCnt%18==0:
            tmpList.append([])
            
        for item in irow[2:11]:
            #print(item)
            if item == "NR":
                tmpList[-1].append(float(0))
            else:
                tmpList[-1].append(float(item))
    
        rowCnt += 1
    
    fptr.close()
    
    testX = np.array(tmpList)
    testX = np.concatenate((testX, np.ones((len(testX),1))), axis=1)
   
   
    ############# Read TestY
    tmpList = []
    fptr  = open(ansFile, "r")
    frow  = csv.reader(fptr, delimiter=",")  
    
    fptr.readline() #skip the first line
    for irow in frow:     
        tmpList.append(float(irow[1]))
    
    testY = np.array(tmpList)
    fptr.close()
    
    ############ Test    
    testR   = np.dot(testX, myModel)
    mse = (testR - testY)**2    
    
    grade_public = np.sqrt(np.mean(mse[:120]))
    grade_private = np.sqrt(np.mean(mse[120:]))
    #grade = np.sqrt(np.sum((diff)**2)/len(diff))
    print("trainLoss = %.4f  RMSE         = %.4f" %(trainLoss, (grade_public+grade_private)/2))
    print("trainLoss = %.4f  RMSE_public  = %.4f" %(trainLoss, grade_public))
    print("trainLoss = %.4f  RMSE_private = %.4f" %(trainLoss, grade_private))
    
    """
    if wen:
        fptr  = open("outfile.csv", "w")
        csvWr = csv.writer(fptr)    
        
        outList = [trainLoss, grade]
        csvWr.writerow([], delimiter=',')

        fptr.close()
       """ 
#end of def test_routing:

    

    
    

