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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import sys

def gen_model(n_users, n_items, latent_dim):
    
    userInputLayer = layers.Input(shape=[1])
    itemInputLayer = layers.Input(shape=[1])
    
    userVec   = layers.Embedding(n_users, latent_dim, embeddings_initializer='random_normal', name='User_Embedding')(userInputLayer)
    userBias  = layers.Embedding(n_users, 1, embeddings_initializer='zeros')(userInputLayer)
    
    userVec   = layers.Flatten()(userVec)    
    userBias  = layers.Flatten()(userBias)    
    
    itemVec   = layers.Embedding(n_items, latent_dim, embeddings_initializer='random_normal', name='Movie_Embedding')(itemInputLayer)
    itemBias  = layers.Embedding(n_items, 1, embeddings_initializer='zeros')(itemInputLayer)
    
    itemVec   = layers.Flatten()(itemVec)    
    itemBias  = layers.Flatten()(itemBias)
    
    r_hat = layers.Dot(name='Dot', axes=1)([userVec, itemVec])
    r_hat = layers.Add(name='Bias')([r_hat, userBias, itemBias])
    
    #outputLayer  = layers.Concatenate()([inputLayer_a, inputLayer_b])
    #keras.layers.Concatenate(axis=-1)
    
    
    model   = models.Model(inputs=[userInputLayer, itemInputLayer], outputs=r_hat)
    model.summary()
    model.compile(loss='mse', optimizer='sgd')
    
    plot_model(model, to_file='tmp/model.png', show_shapes=True, show_layer_names=True)  
    
    return model

def main(args):
    
    
    #paraDict = {'latent_dim': args['latent_dim'],
    #            'batch_size': 128}
    print(type(args))
    print(args)
    
    paraDict = args.__dict__
    print(paraDict)
    #sys.exit(0)
    
    trainX     = pd.read_csv(paraDict['train_path'], index_col=0)
    testX      = pd.read_csv(paraDict['test_path'] , index_col=0)    
    userDF     = pd.read_csv(paraDict['user_path'], index_col=None, delimiter='::')    
    movieDF    = pd.read_csv(paraDict['movie_path'], index_col=None, delimiter='::')   
    #allX       = pd.concat([trainX, testX], axis=0, join='outer')
    
    #userDict = dict( enumerate(trainX['UserID'].astype('category').cat.categories) )
    #print(trainX['UserID'].astype('category').cat.categories)
    
    testX   = testX.head(1000)
    #trainX = trainX.head(10)
    #trainX = trainX.head(10)
    userLE  = preprocessing.LabelEncoder()
    movieLE = preprocessing.LabelEncoder()
    #trainX['UserIdx']  = userLE.fit(trainX['UserID'])
    #print(userLE.classes_)
    userLE.fit(userDF['UserID'])
    movieLE.fit(movieDF['movieID'])
    
    trainX['UserIdx']  = userLE.transform(trainX['UserID'])
    trainX['MovieIdx'] = movieLE.transform(trainX['MovieID'])
    
    n_user  = userDF.shape[0]
    n_movie = movieDF.shape[0]
    
    #n_user  = len(userDF['UserID'].unique()) + 1  #for new id
    #n_movie = len(trainX['MovieID'].unique()) + 1 #for new id
    
    """
    userDict  = dict(zip(userLE.classes_, userLE.transform(userLE.classes_)))
    movieDict = dict(zip(movieLE.classes_,movieLE.transform(movieLE.classes_)))
      
    
    new_user_idx  = n_user  - 1
    new_movie_idx = n_movie - 1
    """

    #trainX['UserIdx']  = trainX['UserID'].astype('category').cat.codes.astype(np.int64)
    #trainX['MovieIdx'] = trainX['MovieID'].astype('category').cat.codes.astype(np.int64)
    
    #print(trainX['UserID'].astype('category').cat.categories)
    
    #for idx, row in testX.iterrows():
    #    trainX.loc[row['UserID']]
    #for row in testX.head(10).iterrows():
    #    testX[]
    #    print(row)
        #testX.loc['UserIdx']
        #textX.loc['MovieIdx'][]

    #trainX.sort_values(by=['MovieID'], inplace=True)    
    #trainX = trainX.head(1000)
    #trainMovie = trainMovie.head(1000)
    print(trainX.head(20))    
    print("User count  : %d"%n_user)    
    print("Movie count : %d"%n_movie)        
    
    #sys.exit(0)    
    #print(trainX['UserID'].value_counts())
    #print(trainX['MovieID'].value_counts())
    
    #sys.exit()
    
# =============================================================================
#   Train
# =============================================================================
    best_model_path = 'best.hdf5'
    
    myModel = gen_model(n_users=n_user, n_items=n_movie, latent_dim=paraDict['latent_dim'])
    trainX, validX = train_test_split(trainX, test_size=0.1)
    
    myEarlyStop      = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    myModelCheckPoint= callbacks.ModelCheckpoint(filepath=best_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    history = myModel.fit(x=[trainX['UserIdx'].values, trainX['MovieIdx'].values], 
                          y=trainX['Rating'], 
                          validation_data=[[validX['UserIdx'].values, validX['MovieIdx'].values], validX['Rating']],
                          epochs=paraDict['n_epoch'],
                          batch_size=paraDict['batch_size'],
                          callbacks=[myEarlyStop, myModelCheckPoint],
                          shuffle=True,
                          verbose=1,
                          )
# =============================================================================
#   Validation
# =============================================================================        
    myModel = models.load_model(best_model_path)
    
    score = myModel.evaluate(x=[validX['UserIdx'].values, validX['MovieIdx'].values], 
                             y=validX['Rating'],
                             batch_size=paraDict['batch_size'],
                             verbose=1)
    print("==========================================")
    print("Test MSE Loss: {0:.4f}".format(np.sqrt(score)))
    print("==========================================")
    
# =============================================================================
#   Prediction
# =============================================================================     
    #new_id = []
    print("Start Prediction")
    
    #testX['UserIdx']  = pd.Series(0, index=testX.index)
    #testX['MovieIdx'] = pd.Series(0, index=testX.index)    
        
    testX['UserIdx']  = userLE.transform(testX['UserID'])
    testX['MovieIdx'] = movieLE.transform(testX['MovieID'])
    
    """
    newUserIdList  = []
    newMovieIdList = []
    for idx, row in testX.iterrows():
        usr_id   = row['UserID']
        movie_id = row['MovieID']
        
        if usr_id in userDict:
            testX.loc[idx,'UserIdx']  = usr_id            
        else:
            testX.loc[idx,'UserIdx']  = new_user_idx
            newUserIdList.append(usr_id)
        
        if usr_id in userDict:
            testX.loc[idx,'MovieIdx'] = movie_id
        else:
            testX.loc[idx,'MovieIdx'] = new_movie_idx
            newUserIdList.append(movie_id)
            
    newUserIdList  = list(set(newUserIdList))
    newMovieIdList = list(set(newMovieIdList))
    
    print("newUserIdList:\n",newUserIdList)
    print("newMovieIdList:\n",newMovieIdList)
    """    
    predY = myModel.predict(x=[testX['UserIdx'].values, testX['MovieIdx'].values],
                            batch_size=paraDict['batch_size'])
    
    testX['Rating'] = pd.Series(predY.flatten(), index=testX.index)       
    testX['Rating'].to_csv("prediction.csv", header=True)
    
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
        
    myParser.add_argument('--n_epoch',      type=int,    default=50,    dest='n_epoch',     help='n_epoch')    
    myParser.add_argument('--batch_size',   type=int,    default=512,   dest='batch_size',  help='batch_size')    
    myParser.add_argument('--latent_dim',   type=int,    default=8,     dest='latent_dim',   help='latent_dim')        
    
    args = myParser.parse_args()    
    main(args)