# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 09:13:20 2018

@author: yilin9999
"""
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SpatialDropout1D
from keras.utils import plot_model
from keras import optimizers
from keras import regularizers
from keras import callbacks
from keras.preprocessing.sequence import pad_sequences

#from sklearn.model_selection import train_test_split
import numpy as np
import time
import dv_analysis
import text_preprocessing as textp

def create_lstm_dnn(paraDict, EmbeddingMatrix=None):
    hiddenSize = paraDict['neuroCnt']
    
    inputLayer = Input(shape=(paraDict['maxSeqLen'],))
    
    if EmbeddingMatrix is not None:
        x = Embedding(input_dim=paraDict['maxVocabSize'], 
                      weights=[EmbeddingMatrix],
                      trainable=False,
                      output_dim=100)(inputLayer)    
    else:
        x = Embedding(input_dim=paraDict['maxVocabSize'], 
                      output_dim=128)(inputLayer)    
    
    x = GRU(hiddenSize, return_sequences=True,  dropout=0.2, recurrent_dropout=0.2)(x)
    x = GRU(hiddenSize, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    #lstmLayer_3 = GRU(hiddenSize, dropout=0.2, recurrent_dropout=0.2)(lstmLayer_2)
  
    for i in range(paraDict['dnnLayers']):       
        x = Dense(units=hiddenSize, 
                  activation='relu',
                  kernel_regularizer=regularizers.l2(paraDict['regularR']))(x)
        x = Dropout(paraDict['dropoutR'])(x)
        hiddenSize = hiddenSize//2
        
    
    outputLayer = Dense(output_dim=1, activation="sigmoid")(x)   
    
    model = Model(inputs=inputLayer, outputs=outputLayer)
    
    model.summary()
    plot_model(model, to_file='%s/model.png' %paraDict['outDir'], show_shapes=True, show_layer_names=True)  

    model.compile(loss='binary_crossentropy',
          optimizer=optimizers.Adam(lr=0.0001),
          metrics=['accuracy'])
    
    return model
    
def create_lstm_dnn2(paraDict):
    
    model = Sequential()
    
    hiddenSize = paraDict['neuroCnt']
    
    model.add(Embedding(input_dim=paraDict['maxVocabSize'], 
                        output_dim=128,
                        input_length=paraDict['maxSeqLen']))
    
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(hiddenSize, dropout=0.2, recurrent_dropout=0.2))    
        
    
    for i in range(paraDict['dnnLayers']):       
        
        model.add(Dense(units=hiddenSize, 
                        activation='relu', 
                        kernel_regularizer=regularizers.l2(paraDict['regularR'])))
        model.add(Dropout(paraDict['dropoutR']))        
        hiddenSize = hiddenSize//2

    
    model.add(Dense(output_dim=1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy',
          optimizer=optimizers.Adam(lr=0.0001),
          metrics=['accuracy'])
    
    model.summary()
    plot_model(model, to_file='%s_model.png' %paraDict['startTime'], show_shapes=True, show_layer_names=True)  
    
    return model
    
def create_dnn(paraDict):
    
    print("Create DNN model")
        
    model = Sequential()
    
    model.add(Dense(input_dim=paraDict['maxVocabSize'], units=paraDict['neuroCnt'], activation='relu'))
    
    for i in range(paraDict['dnnLayers']):
        model.add(Dense(units=paraDict['neuroCnt']//(4*1+i), 
                        activation='relu', 
                        kernel_regularizer=regularizers.l2(paraDict['regularR'])))
        model.add(Dropout(paraDict['dropoutR']))
    
    model.add(Dense(output_dim=1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy',
          optimizer=optimizers.Adam(lr=0.0001),
          metrics=['accuracy'])

    model.summary()
    
    
    return model


def dump_prediction_result(predFile, outputY):
    try:
        with open(predFile, "w") as fptr:
            print("Dump predtion result: %s" %predFile)
            fptr.write("id,label\n")
            
            for i in range(len(outputY)):
                fptr.write("{0},{1}\n".format(i, str(outputY[i])))
                #fptr.write("%d,%s\n" %(i, str(outputY[i])))               
            #np.savetxt(csvFptr, outputY, fmt='%d', delimiter=',')            
        
    except Exception as e:
        print(type(e), str(e))
 

def predict(model, paraDict, docToken, predX, dumpResult=True, pred_class=False):
    
    if paraDict['rmStopwords'] is True:
        predX = textp.remove_stop_word(predX)
        
    #model = model.load_model(modelFile)    
    steps = int(np.ceil(len(predX)/paraDict['batchSize']))
    
    predY = model.predict_generator(generator=generator_batch(predX, None, paraDict, docToken, shuffle=False),
                                    #generator=generator_from_csr_arrays(predX, None, batch_size),
                                    steps=steps,      
                                    max_queue_size=30,                                                                   
                                    verbose=1)
    #predY = model.predict(predX)    
    predY = predY.flatten()
    
    predY       = np.around(predY, decimals=4)
    predY_round = np.around(predY).astype(np.int8)
        
    if dumpResult:
        dump_prediction_result("prediction_org.csv", predY)
        dump_prediction_result("prediction.csv",     predY_round)
    
    if pred_class is True:
        return list(predY_round)
    else:
        return list(predY)
    
       


def generator_batch(X, Y, paraDict, docToken, shuffle=True):
    #print('======= generator initiated =======')   
    
    #sampleCnt        = len(X)
    sampleOfBatchCnt = int(np.ceil(len(X)/paraDict['batchSize']))
    
    batchIdxList = np.arange(sampleOfBatchCnt)
        
    counter = 0     
    if shuffle:
        np.random.shuffle(batchIdxList)
                
    while 1:            
        headIdx = paraDict['batchSize'] * batchIdxList[counter]        
        tailIdx = headIdx+ paraDict['batchSize']                   

        batchX = docToken.transform(X[headIdx:tailIdx])
        if paraDict['algo']!='bow':
            batchX = pad_sequences(batchX, paraDict['maxSeqLen'], padding='post')                        
        
        if Y is not None:
            batchY = np.array(Y[headIdx:tailIdx])            
        
        #print("\nsampleCnt=%d, BatchCnt=%d, counter=%d, batchIdxList[counter]=%d, headIdx=%d, tailIdx=%d" %(sampleCnt, sampleOfBatchCnt, counter, batchIdxList[counter], headIdx, tailIdx))
        #print(X[headIdx:tailIdx])
        #print(batchY)        
        
        sampleOfBatchCnt_m1 = sampleOfBatchCnt-1
        
        if(counter==sampleOfBatchCnt_m1):
            counter = 0
            if shuffle:
                np.random.shuffle(batchIdxList)            
        else:
            counter += 1        
        
        if Y is not None:
            #os.system("pause")
            yield batchX, batchY            
        else:
            yield batchX
    
def train(model, paraDict, docToken, trainX, trainY, validX, validY):

    #np.random.seed(paraDict['seed'])
    #tf.set_random_seed(paraDict['seed'])           
    beginTime = time.time()

    #checkModelName ="%s/{val_acc:.2f}.hdf5" %paraDict['startTime']
    checkModelName ="%s/best.hdf5" %paraDict['outDir']
    
    myEarlyStop       = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
    myModelCheckpoint = callbacks.ModelCheckpoint(checkModelName, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    myTensorBoard     = callbacks.TensorBoard(log_dir='./tboard')


    steps_per_epoch  = int(np.ceil(len(trainX)/paraDict['batchSize']))
    validation_steps = int(np.ceil(len(validX)/paraDict['batchSize']))
    
    #model.reset_states()
    myHistory = model.fit_generator(generator=generator_batch(trainX, trainY, paraDict, docToken, shuffle=True),                        
                                    #generator=generator_from_csr_arrays(trainX, trainY, paraDict['batchSize'], shuffle=True),                        
                                    steps_per_epoch=steps_per_epoch, 
                                    validation_data=generator_batch(validX, validY, paraDict, docToken),
                                    validation_steps=validation_steps,
                                    verbose=1,
                                    max_queue_size=30,
                                    shuffle=False,
                                    epochs=paraDict['epochNum'],
                                    callbacks=[myEarlyStop, myModelCheckpoint, myTensorBoard])    


    endTime = time.time()        
    runtime = endTime-beginTime
    
    dv_analysis.dump_result(myHistory, paraDict, runtime)
    print("Run Time: %.2fsec" %runtime)
    #plot_result(dict(myHistory))    

"""   
def generator_from_csr_arrays(X, Y, batchSize, shuffle=False):
    
    #print('======= generator initiated =======')   
    
    sampleCnt        = X.shape[0]
    sampleOfBatchCnt = int(np.ceil(X.shape[0]/batchSize))
    
    batchIdxList = np.arange(sampleOfBatchCnt)
        
    counter = 0     
    if shuffle:
        np.random.shuffle(batchIdxList)
                
    while 1:            
        headIdx = batchSize * batchIdxList[counter]
        #headIdx = batchSize * counter
        tailIdx = headIdx+batchSize                         
        
        batchX = X[headIdx:tailIdx].toarray()        
        if Y is not None:
            batchY = np.array(Y[headIdx:tailIdx])            
        
        #print("\nsampleCnt=%d, BatchCnt=%d, counter=%d, batchIdxList[counter]=%d, headIdx=%d, tailIdx=%d" %(sampleCnt, sampleOfBatchCnt, counter, batchIdxList[counter], headIdx, tailIdx))
        #print(X[headIdx:tailIdx])
        #print(batchY)        
        sampleOfBatchCnt_m1 = sampleOfBatchCnt-1
        
        if(counter==sampleOfBatchCnt_m1):
            counter = 0
            if shuffle:
                np.random.shuffle(batchIdxList)            
        else:
            counter += 1        
        
        if Y is not None:
            #os.system("pause")
            yield batchX, batchY            
        else:
            yield batchX
"""