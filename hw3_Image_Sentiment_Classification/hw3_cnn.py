# -*- coding: utf-8 -*-
"""
Created on Fri May 25 00:06:25 2018

@author: yilin9999
"""

import numpy as np
from hw3_data import MyData

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras import regularizers

from keras import optimizers
from keras import initializers
from keras import callbacks
from keras.utils.vis_utils import plot_model


from argparse import ArgumentParser

import result_analysis

import time
    
def create_cnn(featureCnt, ClassCnt, cnnParaDict):
    imgShape = (48,48,1) #((batch, height, width, channels))
    kernelSize=(3,3)
    
    cnnModel = Sequential()
    
    for i in range(cnnParaDict['cov2dNum']):
        cnnModel.add(Conv2D(filters= cnnParaDict['filterNum'], 
                     kernel_size   = kernelSize, 
                     input_shape   = imgShape,    #default is channel last
                     padding       = "same",  
                     kernel_initializer ='glorot_normal',
                     activation    ='relu'                   
                     ))    
        cnnModel.add(Conv2D(filters= cnnParaDict['filterNum'], 
                     kernel_size   = kernelSize, 
                     input_shape   = imgShape,    #default is channel last
                     padding       = "same",  
                     kernel_initializer ='glorot_normal',
                     activation    ='relu'                   
                     ))            
        cnnModel.add(MaxPooling2D(pool_size=(2, 2), strides=1))     
    
    #cnnModel.add(Dropout(cnnParaDict['dropoutR']))
    cnnModel.add(Flatten())        
    
    cnnModel.add(Dense(cnnParaDict['neuroCnt'], activation='relu', kernel_regularizer=regularizers.l2(cnnParaDict['regularR'])))    
    cnnModel.add(Dropout(cnnParaDict['dropoutR']))
    
    cnnModel.add(Dense(cnnParaDict['neuroCnt'], activation='relu', kernel_regularizer=regularizers.l2(cnnParaDict['regularR'])))
    cnnModel.add(Dense(output_dim=ClassCnt, activation='softmax'))

    cnnModel.summary()  
    #plot_model(cnnModel, to_file='myModel.h5')
    
    return cnnModel

def main(opts):
    
    trainData = MyData(opts.train_data_path)    
    #trainData = MyData("C:\\testdata\\train.csv")    
    
    trainX, validX, trainY, validY = trainData.split_valid_data(0.2)
    
    featureCnt = trainX.shape[1]
    ClassCnt   = 7
  
    cnnParaDict = {'epochNum' : opts.epochNum,
                   'batchSize': opts.batchSize,
                   'cov2dNum' : opts.cov2dNum,
                   'filterNum': opts.filterNum,
                   'dropoutR' : opts.dropoutR,
                   'neuroCnt' : opts.neuroCnt,
                   'regularR' : opts.regularR
                   }    
    
    cnnModel = create_cnn(featureCnt, ClassCnt, cnnParaDict)
    
    myEarlyStop = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    myOptimizer = optimizers.adam(lr=0.001)
    cnnModel.compile(optimizer=myOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    beginTime = time.time()
    myHistory = cnnModel.fit(trainX, trainY, 
                             epochs=cnnParaDict['epochNum'], 
                             batch_size=cnnParaDict['batchSize'],
                             validation_data=(validX, validY),
                             callbacks=[myEarlyStop],
                             shuffle=True,
                             verbose=1)
    endTime = time.time()
    
    runtime = endTime-beginTime
    print("Run Time: %.2fsec" %runtime)
    
    #MyData.plot_result(myHistory)
    trainData.write_result(myHistory, cnnParaDict, runtime)
       
     #one-hot to class
    validY_class = np.argmax(validY, axis=1)    
    result_analysis.gen_confusion_matrix(model=cnnModel, 
                                         classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"],
                                         validX=validX, 
                                         validY=validY_class, 
                                         paraDict=cnnParaDict, 
                                         dumpfile=1,
                                         plot=1)    
    if opts.plot:
        trainData.read_result()
    
if __name__ == "__main__":
    parser = ArgumentParser(description='CNN')                 
    #group  = parser.add_mutually_exclusive_group() 
    
    parser.add_argument('--train_data_path', 
                        type=str,
                        default="C:\\testdata\\train_small.csv", 
                        dest='train_data_path',
                        help='train_data_path')  
    
    parser.add_argument('--epoch'     ,   type=int,   default=3,    dest='epochNum' , help='epochNum')     
    parser.add_argument('--batch_size',   type=int,   default=300,  dest='batchSize', help='batchSize')     
    parser.add_argument('--cov2d'     ,   type=int,   default=2,    dest='cov2dNum' , help='cov2dNum')     
    parser.add_argument('--filter'    ,   type=int,   default=8,   dest='filterNum', help='filterNum')     
    parser.add_argument('--drop_rate' ,   type=float, default=0,    dest='dropoutR' , help='dropoutR')     
    parser.add_argument('--neuro_cnt' ,   type=int,   default=512,  dest='neuroCnt' , help='neuroCnt')     
    parser.add_argument('--regular_rate', type=float, default=0.01, dest='regularR' , help='regularR')         
    parser.add_argument('-p',    action='store_true', default=False,dest='plot'     , help='plot')     
    
    opts = parser.parse_args()    
    main(opts)