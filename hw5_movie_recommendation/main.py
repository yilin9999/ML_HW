# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 11:52:23 2018

@author: yilin9999
"""
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from keras import models
from keras import layers
from keras import callbacks
#from keras.layers import Input
from keras.utils import plot_model
from keras.constraints import non_neg

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import sys
import time

def gen_model(n_users, n_items, latent_dim):
    
    userInputLayer = layers.Input(shape=[1])
    itemInputLayer = layers.Input(shape=[1])
    
    userVec   = layers.Embedding(n_users, latent_dim, embeddings_initializer='random_normal', name='User_Embedding', embeddings_constraint=non_neg())(userInputLayer)
    userBias  = layers.Embedding(n_users, 1, embeddings_initializer='zeros')(userInputLayer)
    
    userVec   = layers.Flatten()(userVec)    
    userBias  = layers.Flatten()(userBias)    
    
    itemVec   = layers.Embedding(n_items, latent_dim, embeddings_initializer='random_normal', name='Movie_Embedding', embeddings_constraint=non_neg())(itemInputLayer)
    itemBias  = layers.Embedding(n_items, 1, embeddings_initializer='zeros')(itemInputLayer)
    
    itemVec   = layers.Flatten()(itemVec)    
    itemBias  = layers.Flatten()(itemBias)
    
    r_hat = layers.Dot(name='Dot', axes=1)([userVec, itemVec])
    r_hat = layers.Add(name='Bias')([r_hat, userBias, itemBias])
    
    #outputLayer  = layers.Concatenate()([inputLayer_a, inputLayer_b])
    #keras.layers.Concatenate(axis=-1)
    
    
    model   = models.Model(inputs=[userInputLayer, itemInputLayer], outputs=r_hat)
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    
    plot_model(model, to_file='tmp/model.png', show_shapes=True, show_layer_names=True)  
    
    return model

def main(args):
    
        #paraDict = {'latent_dim': args['latent_dim'],
    #            'batch_size': 128}
    print(type(args))
    print(args)
    
    paraDict = args.__dict__
    print(paraDict)
    
    if paraDict['pred_only'] is True:        
        predict(paraDict)        
        sys.exit(1)
        
        
    trainX     = pd.read_csv(paraDict['train_path'], index_col=0)    
    userDF     = pd.read_csv(paraDict['user_path'], index_col=None, delimiter='::')    
    movieDF    = pd.read_csv(paraDict['movie_path'], index_col=None, delimiter='::')   
    #allX       = pd.concat([trainX, testX], axis=0, join='outer')
    
    #userDict = dict( enumerate(trainX['UserID'].astype('category').cat.categories) )
    #print(trainX['UserID'].astype('category').cat.categories)
    
    userLE, movieLE = get_usr_movie_idx(paraDict['user_path'], paraDict['movie_path'])
    
    trainX['UserIdx']  = userLE.transform(trainX['UserID'])
    trainX['MovieIdx'] = movieLE.transform(trainX['MovieID'])
    
    n_user  = userDF.shape[0]
    n_movie = movieDF.shape[0]
    
    #n_user  = len(userDF['UserID'].unique()) + 1  #for new id
    #n_movie = len(trainX['MovieID'].unique()) + 1 #for new id
    
    """
    userDict  = dict(zip(userLE.classes_, userLE.transform(userLE.classes_)))
    movieDict = dict(zip(movieLE.classes_,movieLE.transform(movieLE.classes_)))        
    """
    #trainX['UserIdx']  = trainX['UserID'].astype('category').cat.codes.astype(np.int64)
    #trainX['MovieIdx'] = trainX['MovieID'].astype('category').cat.codes.astype(np.int64)
    
    print(trainX.head(20))    
    print("User count  : %d"%n_user)    
    print("Movie count : %d"%n_movie)        
    
# =============================================================================
#   Train
# =============================================================================
    begin_time = time.time()
    myModel = gen_model(n_users=n_user, n_items=n_movie, latent_dim=paraDict['latent_dim'])
    trainX, validX = train_test_split(trainX, test_size=0.1)
    
    myEarlyStop      = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    myModelCheckPoint= callbacks.ModelCheckpoint(filepath=paraDict['output_model_path'], monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    history = myModel.fit(x=[trainX['UserIdx'].values, trainX['MovieIdx'].values], 
                          y=trainX['Rating'], 
                          validation_data=[[validX['UserIdx'].values, validX['MovieIdx'].values], validX['Rating']],
                          epochs=paraDict['n_epoch'],
                          batch_size=paraDict['batch_size'],
                          callbacks=[myEarlyStop, myModelCheckPoint],
                          shuffle=True,
                          verbose=1,
                          )
    
    historyDF = pd.DataFrame(index=np.arange(len(history.history['val_loss'])))
    historyDF['val_loss'] = pd.Series(history.history['val_loss'])
    historyDF.to_csv("history.csv")
    
    end_time = time.time()
# =============================================================================
#   Validation
# =============================================================================        
    myModel = models.load_model(paraDict['output_model_path'])    
    
    score = myModel.evaluate(x=[validX['UserIdx'].values, validX['MovieIdx'].values], 
                             y=validX['Rating'],
                             batch_size=paraDict['batch_size'],
                             verbose=1)
    print("==========================================")
    print("Test MSE Loss: {0:.4f}".format(np.sqrt(score)))
    print("==========================================")
    print("Training Time: {0:d} sec".format(int(end_time-begin_time)))
    predict(paraDict)
    
# =============================================================================
#   Prediction
# =============================================================================     
def predict(paraDict):
    model = models.load_model(paraDict['output_model_path'])    
    
    print("Start Prediction")        
    
    testX      = pd.read_csv(paraDict['test_path'] , index_col=0)        
    userLE, movieLE = get_usr_movie_idx(paraDict['user_path'], paraDict['movie_path'])        
    
    testX['UserIdx']  = userLE.transform(testX['UserID'])
    testX['MovieIdx'] = movieLE.transform(testX['MovieID'])
 

    predY = model.predict(x=[testX['UserIdx'].values, testX['MovieIdx'].values],
                          batch_size=paraDict['batch_size'],
                          verbose=1)
    
    
    testX['Rating'] = pd.Series(predY.flatten(), index=testX.index)       
    testX['Rating'].to_csv("prediction.csv", header=True)  
    print("Dump prediction.csv")   
    

def get_usr_movie_idx(user_path, movie_path):
        userDF     = pd.read_csv(user_path, index_col=None, delimiter="::")    
        movieDF    = pd.read_csv(movie_path, index_col=None, delimiter="::")   
        userLE  = preprocessing.LabelEncoder()
        movieLE = preprocessing.LabelEncoder()    
        userLE.fit(userDF['UserID'])
        movieLE.fit(movieDF['movieID'])
        
        return userLE, movieLE
# =============================================================================
#   Main  
# =============================================================================
if __name__ == "__main__":
    myParser = ArgumentParser("movie_recommendation")
    
    myParser.add_argument('--train_path', 
                          type=str,                          
                          default="data/train.csv",                       
                          dest='train_path',
                          help='train_path') 
    
    myParser.add_argument('--test_path', 
                          type=str,                          
                          default="data/test.csv",                       
                          dest='test_path',
                          help='test_path') 
    
    myParser.add_argument('--user_path', 
                          type=str,                          
                          default="data/users.csv",                       
                          dest='user_path',
                          help='user_path') 
    
    myParser.add_argument('--movie_path', 
                          type=str,                          
                          default="data/movies.csv",                       
                          dest='movie_path',
                          help='movie_path') 
    
    myParser.add_argument('--output_model_path', 
                          type=str,                          
                          default="best.hdf5",                       
                          dest='output_model_path',
                          help='output_model_path') 
    
    myParser.add_argument('-pred_only',     action='store_true',       dest='pred_only',     help='pred_only')        
    #myParser.add_argument('-pred_only',     action='store_false',       dest='pred_only',     help='pred_only')        
    myParser.add_argument('--n_epoch',      type=int,    default=1,    dest='n_epoch',     help='n_epoch')    
    myParser.add_argument('--batch_size',   type=int,    default=512,   dest='batch_size',  help='batch_size')    
    myParser.add_argument('--latent_dim',   type=int,    default=8,     dest='latent_dim',   help='latent_dim')        
    
    args = myParser.parse_args()    
    main(args)