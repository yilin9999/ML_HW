# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:43:07 2018

@author: yilin9999
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_validate    
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score
import sys

def just_fit(modelType, model, trainX, trainY, validX, validY):
    if modelType=='rf':
        model.fit(trainX, trainY)  
    elif modelType == 'xgb':            
        model.fit(trainX, trainY,  early_stopping_rounds=100, eval_metric="error", eval_set=[(validX, validY)], verbose=False)
    else:
        raise ValueError('Wrong model type');    
    
    predYk = model.predict(validX)    
    
    score_acc = accuracy_score(validY, predYk)    
    score_f1  = f1_score(validY, predYk, average='binary') 
    
    if hasattr(model, 'oob_score_'):
        score_train = model.oob_score_
    else:
        score_train = 1 - model.best_score            
        
    return score_train, score_acc, score_f1

def final_cross_validate(modelType, model, trainX, trainY, cvNum, random_state, scoring=['f1', 'accuracy']):
    
    skFold = StratifiedKFold(n_splits=cvNum, shuffle=False)
    #skFold  = KFold(n_splits=cvNum, shuffle=False)    
    
    index      = 0
    min_acc    = 1
    #min_validXk= None
    #min_validYk= None
    

    scoreTable   = pd.DataFrame(np.nan, index=range(1, cvNum+1), columns=['train_accuracy', 'test_accuracy', 'test_f1'])
    feImportance = pd.DataFrame(np.nan, index=range(1, cvNum+1), columns=trainX.columns)
    #d = pd.DataFrame(0, index=np.arange(len(data)), columns=feature_list)
    
    for idxT, idxV in skFold.split(trainX, trainY):
        
        #print(trainX.iloc[idxT])
        #print(trainY.iloc[idxT])            
        index += 1
        print("Cross Valdation {0}/{1}".format(index, cvNum))
        
        
        trainXk, trainYk = trainX.iloc[idxT], trainY.iloc[idxT]
        validXk, validYk = trainX.iloc[idxV], trainY.iloc[idxV]
        
        score_train, score_acc, score_f1 = just_fit(modelType, model, trainXk, trainYk, validXk, validYk)
        #if modelType=='rf':
        #    model.fit(trainXk, trainYk)  
        #elif modelType == 'xgb':            
        #    model.fit(trainXk, trainYk,  early_stopping_rounds=100, eval_metric="error", eval_set=[(validXk, validYk)], verbose=False)
        #else:
        #    raise ValueError('Wrong model type');            
        
        #score    = modelRF.score(validXk, validYk)                    
        #predYk = model.predict(validXk)                                
        
        scoreTable.loc[index, 'train_accuracy'] = score_train
        scoreTable.loc[index, 'test_accuracy']  = score_acc
        scoreTable.loc[index, 'test_f1']        = score_f1
                        
        feImportance.loc[index] = model.feature_importances_
        
        if(scoreTable.loc[index, 'test_f1'] < min_acc):
            min_acc = scoreTable.loc[index, 'test_f1']
            
            #trainXk_worst = pd.concat([trainX.iloc[idxT], trainY.iloc[idxT]], axis=1)
            #validXk_worst = pd.concat([trainX.iloc[idxV], trainY.iloc[idxV]], axis=1)
            
            pd.Series(trainX.iloc[idxT].index).to_csv("tmp/trainXk_worst_set.csv", index=False)
            pd.Series(trainX.iloc[idxV].index).to_csv("tmp/validXk_worst_set.csv", index=False)
            #trainXk_worst.to_csv("tmp/trainXk_worst_f1_{0:.4f}.csv".format(min_acc))
            #validXk_worst.to_csv("tmp/validXk_worst_f1_{0:.4f}.csv".format(min_acc))                
            #min_f1       = scoreTable.loc[index, 'test_f1']
            #min_validIdx = idxV
            #min_trainIdx = idxT
            #mismatchIdx = validYk[predYk != validYk].index
        #print("Feature Counts = {0}".format(modelRF.n_features_ ))    
        #print("[Fold {0}] RF score: {1:>8.4f} valid score: {2:>8.4f}".format(counter, oobScore, score))
        
        
    #else:
    #    scorDict = cross_validate(model, trainX, trainY, cv=cvNum, scoring=scoring, verbose=2)    
    #    scoreTable = pd.DataFrame.from_dict(scorDict).round(4)            
    #    print(scoreTable)
    
    scoreSummary = scoreTable.agg(['mean', 'std'])
    #scoreTable.loc['std']  = scoreTable.iloc[0:cvNum].std()
    #scoreTable.loc['mean'] = scoreTable.iloc[0:cvNum].mean()
    feImportanceSummary = feImportance.agg(['mean','std'])
    #feImportance.loc['std']  = feImportance.iloc[0:cvNum].std()
    #feImportance.loc['mean'] = feImportance.iloc[0:cvNum].mean()
        
    print(scoreTable)
    #print(scoreSummary)    
    print(feImportance)
    #print(feImportanceSummary)
    
    
    print("==================================")    
    for col in feImportanceSummary:        
        print('{0:<8}: {1:>8.4f} (+/-{2:>8.4f})'.format(col, feImportanceSummary.loc['mean', col], feImportanceSummary.loc['std', col]))
    print("==================================")    
    print("TACC score :{0:.4f}, (+/-{1:.4f})".format(scoreSummary.loc['mean', 'train_accuracy'], scoreSummary.loc['std', 'train_accuracy']*2))
    print("VACC score :{0:.4f}, (+/-{1:.4f})".format(scoreSummary.loc['mean', 'test_accuracy'] , scoreSummary.loc['std', 'test_accuracy']*2))
    print("F1   score :{0:.4f}, (+/-{1:.4f})".format(scoreSummary.loc['mean', 'test_f1']       , scoreSummary.loc['std', 'test_f1']*2))            
    print("==================================")
    
    #sys.exit(0)
    #print(min_acc)
    #print(min_validXk)
    
    #FinalScore = scoreTable.loc['mean':]    
    return scoreSummary, feImportanceSummary, min_acc

def train(model, trainX, trainY):
    
    model.fit(trainX, trainY)
    
    oobScore     = model.oob_score_
    importances  = model.feature_importances_
    
    feImportance = dict(zip(trainX.columns, importances))
    print("====================================")    
    #for itr in feImportance:
    for key, value in feImportance.items():
        print("{0:<8}: {1:.4f} ({2})".format(key, value, trainX[key].dtype))
    
    print("\nOOB score: {0:.4f}".format(oobScore))
    print("====================================")
    
