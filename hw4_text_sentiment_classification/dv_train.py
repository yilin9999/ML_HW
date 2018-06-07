# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:15:41 2018

@author: yilin9999
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import optimizers
from keras import callbacks
import time
                         
                          
def plot_result(readDict):
    plt.subplot(211)  
    plt.plot(readDict['acc'])
    plt.plot(readDict['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    
    plt.subplot(212)  
    # summarize history for loss
    plt.plot(readDict['loss'])
    plt.plot(readDict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig('train_result.png',dpi=600)
    plt.show()
    

def train(trainX, trainY, model, paraDict):
    np.random.seed(paraDict['seed'])
    tf.set_random_seed(paraDict['seed'])    
    
    print(paraDict)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001),
                  metrics=['accuracy']
                  )
    
    myEarlyStop = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto')
    
    beginTime = time.time()
    myHistory = model.fit(x=trainX, y=trainY,
                          epochs=paraDict['epochNum'],
                          batch_size=paraDict['batchSize'],                          
                          validation_split=0.1,
                          shuffle=False,
                          verbose=2,
                          callbacks=[myEarlyStop]
                          )
    endTime = time.time()
    
    runtime = endTime-beginTime
    print("Run Time: %.2fsec" %runtime)
    
    #plot_result(dict(myHistory))
