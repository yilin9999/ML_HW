# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:27:49 2018

@author: yilin9999
"""

import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import sys
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from train import final_cross_validate
from train import train
from train import just_fit

from sklearn.utils import shuffle
from data_analysis import analysis
from data_analysis import TITLE_SET, PCLASS_SET

#from sklearn.ensemble import RandomForestRegressor

from numpy import int64
#from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer


class InputData():
    def __init__(self, filePath):
        #with open(filePath) as ftpr:
        #self.df = pd.DataFrame.from_csv(filePath, index_col=0)        
        self.df = pd.read_csv(filePath, index_col=0)  
    #def check_:

def fillna_fare(DF):    
    
    print("======== Pclass value counts ========")
    print(DF['Pclass'].value_counts())
    
    print('Filling NaN Fare by median')
    print(DF[DF['Fare'].isnull()]['Fare'])
        
    for i in range(1,3 +1):
        flt = DF['Pclass']==i
        median = DF[flt]['Fare'].median()
        #print(allDF[flt][['Fare', 'Pclass']].loc[1040:1050])
        #print("Pclass {0} Fill median:{1}".format(i ,median))        
        allDF.update(DF[flt]['Fare'].fillna(median))               
        #print(allDF[flt][['Fare', 'Pclass']].loc[1040:1050])
        
        
def fillna_age(DF):       
    #avgAgeDict = {}
    
    Title = DF['Title'].value_counts().index.tolist()
    #print(Title)
    #sys.exit(0)
    for title in Title:
        flt  = DF['Title']==title
        mean = round(DF[flt]['Age'].mean())
        DF.update(DF[flt]['Age'].fillna(mean))
        print("avg age in {0}: {1}".format(title, mean))
    
    #DF.update(DF['Age'].fillna(DF['Age'].mean()))
    
    
    #Title vs. Survived
    #plt.figure()   
    #sns.countplot(DF['Pclass'], hue=DF['Age'])
    #sns.countplot(DF['Age'], hue=DF['Pclass'])
    
    # Age vs. Fare
    #plt.figure()       
    #print(DF[DF['Survived']==0]['Survived'])
    #sns.displot(DF['Age'], hue=DF[DF['Survived']==0]['Survived'])        
    #sns.jointplot(x='Age', y='Fare', data=DF, kind='reg')
    
    #sys.exit(0)
    
def fillna_embarked(DF):
    print(DF['Embarked'].fillna('S', inplace=True))       


def create_feature_agerange(DF):    
    
    qc = 7
    meanF = DF['Age'].mean()
    diffF = DF['Age'].max() - DF['Age'].min()
    varF  = DF['Age'].var()
    
    sF =DF['Age'].apply(lambda x: (x - meanF) / varF)    
    
    ageRange      = pd.qcut(DF['Age'],q=qc)
    #ageRange      = pd.qcut(sF,q=qc)
    gp_agerange   = ageRange.groupby(ageRange)
    
    #print(gp_agerange.ngroup())
    
    #sns.countplot(DF['Survived'], hue=gp_agerange.ngroup())
    #sns.countplot(DF['Survived'], hue=gp_agerange.ngroup())
    
    #print(gp_agerange.size())
    #sys.exit(0)
    return gp_agerange.ngroup()
    
def create_feature_fcrange(DF):    
    
    #min_max_scaler  = preprocessing.MinMaxScaler()
    #tt = min_max_scaler.fit_transform(DF['Fare'])
    #print(tt)
    meanF = DF['Newfare'].mean()
    diffF = DF['Newfare'].max() - DF['Newfare'].min()
    varF  = DF['Newfare'].var()
    
    
    nF =DF['Newfare'].apply(lambda x: (x - meanF) / diffF)
    sF =DF['Newfare'].apply(lambda x: (x - meanF) / varF)
    
    # =DF['Fare'].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    
    #print(n)
    qc = 6
    fareRange   = pd.qcut(DF['Newfare'],q=qc)
    fareRange_n = pd.qcut(nF,q=qc)
    fareRange_s = pd.qcut(sF,q=qc)
            
    gp_Frange   = fareRange.groupby(fareRange)
    gp_Frange_n = fareRange.groupby(fareRange_n)
    gp_Frange_s = fareRange.groupby(fareRange_s)
    
    """
    print(pd.concat([gp_Frange.ngroup(), DF['Fare']], axis=1))
    
    plt.figure()
    #sns.distplot(DF['Fare'], label='orginal', kde=True, hist=False)    
    sns.distplot(nF, label='min_max scale', kde=True, hist=False)        
    sns.distplot(sF, label='normalize', kde=True, hist=False)    
    plt.figure()
    sns.distplot(gp_Frange.ngroup(), label='orginal', kde=True, hist=False)    
    sns.distplot(gp_Frange_n.ngroup(), label='min_max scale', kde=True, hist=False)    
    sns.distplot(gp_Frange_s.ngroup(), label='normalize', kde=True, hist=False)
    sys.exit(0)
    """
    
    return gp_Frange.ngroup()
    #return gp_Frange_n.ngroup()
    #return gp_Frange_s.ngroup()
    
    
def create_feature_family(DF):
    s = pd.Series(DF['SibSp'] + DF['Parch'], dtype='category')
    return s

def create_feature_tconnect(DF):
    gp_Ticket = DF.groupby(['Ticket'])        
    #gpn_Ticket = gp_Ticket.ngroup()    
    Tconnect = pd.Series(0, index=DF.index)
    
    gIdx = 1    
    for name, group in gp_Ticket:
        if(group.shape[0]>1):  #take signle ticket as the same group
    #        print(group.index)
            for idx in group.index:
                Tconnect[idx] = gIdx
            gIdx += 1                
                
    #print(GTicket)
    #return gp_Ticket.ngroup()
    return Tconnect

def create_feature_tgroup(DF, merge_with_family=True):
    gp_Ticket = DF.groupby(['Ticket'])        
    
    TGroup   = pd.Series(index=DF.index)    
    for name, group in gp_Ticket:            
        for row in group.index:            
            TGroup[row] = group.shape[0]
    
    #merge Family and TGroup    
    
    if merge_with_family==True:
        TGroup = TGroup
        for idx in DF.index:        
            TGroup[idx] = max(TGroup[idx], DF['Family'][idx])
    
    #sns.distplot(TGroup, kde=True, label='TGroup')
    #sns.distplot(DF['Family'], kde=True, label='Family')
    
    return TGroup

def create_featyre_ncabin(DF):
    NCabin = pd.Series(index=DF.index, dtype=np.int8)
    for idx, value in DF['Cabin'].iteritems():        
        if value is np.nan:            
            NCabin[idx] = 0
        else:
            NCabin[idx] = 1
    
    return NCabin
    
def create_feature_title(DF):
    
    ##################################
    ## Analyze Name
    ##################################
    nameList = allDF['Name'].tolist()
    
    ctok = Tokenizer()
    ctok.fit_on_texts(nameList)
    
    tokFreq = ctok.word_counts    
    tokFreq = sorted(tokFreq.items(), key=lambda x: x[1], reverse=True)    
    for i in range(10):
        print(tokFreq[i])
        
    
    ##################################
    ## Create Feature
    ##################################         
    #titleSet = ('william', 'john')    
    
    titleList = []
    for idx, tmp in DF.iterrows():           
        name = tmp['Name'].lower()        
        sex  = tmp['Sex']        
        age  = tmp['Age']        
        family = tmp['Family']        
        #print(name)
        tok = re.split('[^\w]+', name)
        #print(tok)        
        for itr in tok:
            if itr in TITLE_SET:                    
                if age <= 5:
                    titleList.append('baby')                    
                #elif (age > 45) & (sex=='male'):
                #    titleList.append('child')                    
                #elif (age > 45) & (sex=='female'):
                #    titleList.append('girl')                    
                else:
                    titleList.append(itr)
                """
                elif age < 15:
                    if sex=='female':
                    #print (tmp['Survived', 'Pclass'])
                        titleList.append('girl')                      
                    else:
                        #print(tmp['Name'])
                        titleList.append('master')                          
                    #titleList.append(itr)                      
                elif age<35 and sex=='male' and family==0:
                    titleList.append('strong')                                              
                """
                
                
                #titleList.append(itr)                      
       #         print(itr)
                break
        else:
            if sex == 'male':
                titleList.append('mr')
            elif sex=='female':
                if age < 15:                
                    titleList.append('miss')
                else:
                    titleList.append('mrs')
            else:
                raise ValueError('Cannont judge Title by both name and sex')
        
    #sys.exit(0)    
    #sns.countplot(DF['Title'], hue=DF['Sex'])
    #ft =  (allDF['Title']=='mr')  & (allDF['Sex']=='female')
    #print(allDF[ft][['Title', 'Sex']])    
    
    
    assert len(titleList) == len(DF['Name'])    
    s = pd.Series(titleList, dtype='category', index=allDF.index)    
    #print(titleList[:10])
    #print(s.head(10))        
    return s

def create_newfare(DF):
    Newfare = pd.Series(index=DF.index)
    
    Newfare = DF['Fare']/DF['TGroup']
    return Newfare
    
def create_nacount(DF):
            
    Nacount = pd.Series(index=DF.index)
    
    #for row in DF.head(10).iterrows():
    for idx, row in DF.drop(['Survived'], axis=1).iterrows(): 
        #print(idx, row)        
        count = 0
        for itr in row:                        
            if pd.isnull(itr):
                count += 1
        Nacount[idx] = count
        
    #print(Nacount)
    #t = Nacount.groupby(Nacount)
    #print(t.size())
    #sys.exit(0)
    return Nacount
    
def gen_train_data(rawTrainDF, onehotCate):
    trainX = pd.DataFrame(index=rawTrainDF.index)
    #trainY = pd.DataFrame(index=rawTrainDF.index)
    
    #print(rawTrainDF.dtypes)
    
    #CategoricalFeatures  = ('Title', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked')
    CategoricalFeatures  = ['Title',                             
                            'Pclass',
                            'SibSp', 
                            'Parch', 
                            'Embarked',
                            'Tconnect',
                            'NCabin',                                                                                                                                                              
                            'Nacount',      
                            'Family',   
                            'TGroup', 
                            'Sex' 
                            ]
    ContiniousFeatures   = ['Age',    
                            'ACrange',
                            'FCrange',   
                            'Newfare',
                            'Fare']
    #ContiniousFeatures   = ['Age', 'FCrange']
    CategoricalNames = {}
    
    for itr in CategoricalFeatures:
        le = preprocessing.LabelEncoder()
        trainX[itr] = le.fit_transform(rawTrainDF[itr])
        CategoricalNames[itr] = le.classes_
    
    if onehotCate is True:    
        for itr in CategoricalFeatures:    
            onehotDF = pd.get_dummies(trainX[itr], prefix=itr).astype(np.int8)
            trainX = pd.concat([trainX, onehotDF], axis=1)        
            trainX.drop(columns=[itr], inplace=True)
    
    for itr in ContiniousFeatures:
        trainX[itr] = rawTrainDF[itr].astype(np.float64)

    """ 
    for col in TRAIN_CATE_SET:
        print(col)
        trainX[col] = rawTrainDF[col].astype(int, copy=True)
    
    for col in TRAIN_CONT_SET:
        trainX[col] = rawTrainDF[col].astype(np.float64, copy=True)
    """
    
    #print(trainX)
    #print(trainX.dtypes)
    #print(trainY.dtypes)
    #print(CategoricalNames)
    return trainX

    
if __name__ == '__main__':    
    #InputData("./data/train.csv")
    trainDF = pd.read_csv("./data/train.csv", index_col=0)        
    testDF  = pd.read_csv("./data/test.csv", index_col=0)        
    
    #validCheck = False
# =============================================================================
#   Parameters
# =============================================================================
    #onehotCate = True        
    modelType  = 'xgb'
        
    featureGroup = []
    parameterGroup = []    
    
    #featureGroup.append(['Pclass', 'Sex', 'Fare', 'Title', 'Family'])     
    #featureGroup.append(['Pclass', 'Sex', 'Newfare', 'Title', 'TGroup'])     
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', ])     
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'Family', 'TGroup', 'Fare'])     
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'Family', 'TGroup', 'Newfare'])     
    #featureGroup.append(['Pclass', 'Sex', 'Title', 'Family', 'TGroup', 'Newfare'])     
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', 'Newfare'])
    #featureGroup.append(['Pclass', 'Sex', 'ACrange', 'Title', 'TGroup', 'Newfare'])     
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'Family'])
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup'])
    
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', 'FCrange', 'Tconnect'])
    #featureGroup.append(['Sex', 'Age', 'Title', 'TGroup', 'FCrange','Tconnect'])
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', 'Newfare', 'Tconnect'])
    featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', 'Newfare'])
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', 'Tconnect'])
    #featureGroup.append(['Sex', 'Age', 'Title', 'TGroup', 'Newfare','Tconnect'])
    
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', 'Newfare'])
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', 'Newfare', 'Tconnect', 'NCabin'])
    
    #featureGroup.append(['Pclass', 'Sex', 'ACrange', 'Title', 'Family'])
    #featureGroup.append(['Pclass', 'Sex', 'ACrange', 'Title', 'TGroup'])
    
    #featureGroup.append(['Pclass', 'Sex', 'Title', 'TGroup'])     
    
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'Family', 'TGroup', 'Newfare'])     
    
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Title', 'TGroup', 'Newfare'])     
    #featureGroup.append(['Pclass', 'Sex', 'Fare', 'Title', 'Family'])     
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'TGroup'])   
    #featureGroup.append(['Pclass', 'Sex', 'Fare', 'Title', 'TGroup'])     
    #featureGroup.append(['Pclass', 'Sex', 'Fare', 'Title', 'TGroup'])     
    #featureGroup.append(['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'Family', 'Tconnect', 'TGroup', 'NCabin'])      
    #featureGroup.append(['Pclass', 'Sex', 'ACrange', 'Fare', 'Title', 'Family', 'Tconnect', 'TGroup', 'NCabin'])          
    #featureGroup.append(['Pclass', 'Sex', 'ACrange', 'Fare', 'Title', 'Tconnect', 'TGroup', 'NCabin'])  
    
    #parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':3, 'cvNum':10})
    #parameterGroup.append({'vRound':  30, 'Nestimators':150, 'MaxDepth':3, 'cvNum':8})
    #parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':3, 'cvNum':7})
    #parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':3, 'cvNum':6})
    #parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':3, 'cvNum':5})
    #parameterGroup.append({'vRound': 30, 'Nestimators':500, 'MaxDepth':3, 'cvNum':4})
    #parameterGroup.append({'vRound': 30, 'Nestimators':400, 'MaxDepth':3, 'cvNum':4})
    #parameterGroup.append({'vRound': 30, 'Nestimators':300, 'MaxDepth':3, 'cvNum':4})
    #parameterGroup.append({'vRound': 30, 'Nestimators':200, 'MaxDepth':3, 'cvNum':4})
    #parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':2, 'cvNum':4})
    
    parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':3, 'subSample':0.8, 'cvNum':4})
    
    
    
    #parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':4, 'cvNum':4})
    #parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':5, 'cvNum':4})
    #parameterGroup.append({'vRound': 30, 'Nestimators':150, 'MaxDepth':6, 'cvNum':4})
    
           
    hyperParGroup = list(itertools.product(featureGroup, parameterGroup))
    
#    validCheck = False
    ranState  = 1
    onehotCate = False
    
# =============================================================================
#   Clean Feature
# =============================================================================
    
    allDF = pd.concat([trainDF, testDF], axis=0)
    allDF.info()
    
    allDF['Family']   = create_feature_family(allDF)
    allDF['Title']    = create_feature_title(allDF)
    allDF['Tconnect'] = create_feature_tconnect(allDF)
    allDF['TGroup'] = create_feature_tgroup(allDF)
    allDF['NCabin'] = create_featyre_ncabin(allDF)
    allDF['Nacount'] = create_nacount(allDF)
    
    ## Check Nan values
    trainNanSum = trainDF.isnull().sum()
    testNanSum  = testDF.isnull().sum()
    
    allNanSum = allDF.isnull().sum()
    print(allNanSum)  
    #print("Total sample: {0}".format(allNanSum.sum()))
    
    #allDF.to_csv("data/oallDF.csv")
    #analysis(allDF)
    fillna_fare(allDF)
    fillna_age(allDF)
    fillna_embarked(allDF)    
    allDF['ACrange'] = create_feature_agerange(allDF)
    allDF['Newfare'] = create_newfare(allDF)
    allDF['FCrange'] = create_feature_fcrange(allDF)
    allNanSum = allDF.isnull().sum()    
    print(allNanSum)
    
    #trainX = allDF[:10]
    print(trainDF.shape)
    print(testDF.shape)
    
    allDF.to_csv("data/allDF.csv")
    #with open("allDF.pkl","wb") as ftpr:
    #pickle.dump(allDF, )

# =============================================================================
#   Prepare TrainX, TrainY
# =============================================================================  
    
    rawTrainDF = allDF[:trainDF.shape[0]]
    trainX = gen_train_data(rawTrainDF, onehotCate=onehotCate)    
    trainY = rawTrainDF['Survived'].astype(int)
  

# =============================================================================
#   Cross Validation
# =============================================================================
    fIdx           = 1    
    best_score     = 0    
    worst_score    = 1
    bestImpSummary, worstImpSummary      = None, None
    bestHyperParGroup,worstHyperParGroup = None, None
    
    scoreColumns = ['train_accuracy', 'test_accuracy', 'test_f1', 'test_f1_2std']
    finalScore = pd.DataFrame(np.nan, columns=scoreColumns,
                              index=np.arange(1, len(hyperParGroup)+1))    
    
    print(len(hyperParGroup))
    
    for fi in range(len(hyperParGroup)):        
        featureList = hyperParGroup[fi][0]
        parList     = hyperParGroup[fi][1]               
        
        scoreDict = {}
        for itr in scoreColumns:
            scoreDict[itr] = 0
        
        
        for i in range(parList['vRound']):                                
            trainX, trainY = shuffle(trainX, trainY, random_state=i+10)
            
            if modelType == 'rf':                
                modelRF = RandomForestClassifier(n_estimators=parList['Nestimators'],
                                 bootstrap=True,                                     
                                 criterion='gini',
                                 max_depth=parList['MaxDepth'],
                                 max_features='auto',
                                 #min_samples_split=,
                                 #min_samples_leaf=1,
                                 #min_weight_fraction_leaf=0.3,
                                 max_leaf_nodes=10, 
                                 oob_score=True,
                                 random_state=randState,
                                 n_jobs=1,
                                 verbose=0)    
                        
            elif modelType == 'xgb':
                modelRF = XGBClassifier(n_estimators=parList['Nestimators'], 
                                        max_depth=parList['MaxDepth'], 
                                        eta=0.1,subsample=parList['subSample'], 
                                        randdom_state=ranState)
            else:
                raise ValueError("Wrong modelType")
                
            scoreSummary, impSummary, min_acc = final_cross_validate(modelType, modelRF, trainX[featureList], trainY, cvNum=parList['cvNum'], random_state=ranState, scoring=['f1','accuracy'])
            
            if hasattr(modelRF, 'oob_score_'):
                print("Best score: {0}: {1:.4f}".format(i, modelRF.oob_score_))
            else:
                print("Best score: {0}: {1:.4f}".format(i, 1 - modelRF.best_score))
            
            #finalScore = pd.concat([finalScore, score], axis=0)                 
            
            scoreDict['train_accuracy'] += scoreSummary.loc['mean', 'train_accuracy']
            scoreDict['test_accuracy']  += scoreSummary.loc['mean', 'test_accuracy']
            scoreDict['test_f1']        += scoreSummary.loc['mean', 'test_f1']            
            scoreDict['test_f1_2std']   += scoreSummary.loc['std',  'test_f1'] * 2            

            #if scoreSummary.loc['mean','test_f1'] > best_score:
            if scoreDict['test_f1'] > best_score:
                best_score         = scoreSummary.loc['mean', 'test_f1']
                #best_score_idx[fi] = i
                #bestModel          = modelRF
                bestImpSummary     = impSummary
                bestHyperParGroup  = hyperParGroup[fi]
                
                
                #(bestTrainX, bestTrainY) = trainX[featureList], trainY
                #(bestvalidX, bestvalidY) = validX[featureList], validY
            
            if scoreDict['test_f1'] < worst_score:
                worst_score         = scoreDict['test_f1']
                #worstModel          = modelRF                                
                worstImpSummary     = impSummary
                worstHyperParGroup  = hyperParGroup[fi]
                
            
            #finalScore.loc[fIdx:] = i
            #pd.concat([finalScore, score], axis=0)                
        
        fIdx = fi + 1
        #finalScore['Group'][fIdx] = fIdx
        for key, value in scoreDict.items():
            finalScore[key][fIdx] = round(value/parList['vRound'],4)
    
    finalSummary = finalScore.agg(['mean', 'std'])
    #print(finalScore)    
    #worstDF = pd.concat([trainY.loc[worstIdx], trainX.loc[worstIdx]], axis=1)
    #finalScore.set_index(np.arange(0, len(finalScore)), drop=True, inplace=True)        
    #best_score      = finalScore.loc[finalScore['test_f1'] == finalScore['test_f1'].max()]
    #bi = best_score.index(max(best_score))
    #bestFeatureList = hyperParGroup[bi][0]
    #bestparList     = hyperParGroup[bi][1]
    #bestModel       = bestModel[bi]
    
    print(bestHyperParGroup)
    print(worstHyperParGroup)
    

                
    print("Best  Model F1 score: {0:.4f}".format(best_score))        
    print("Worst Model F1 score: {0:.4f}".format(worst_score))    
    print("=========================================================================")
    print("Feature Impoertance")
    for col in bestImpSummary:        
        print('{0:<8}: {1:>8.4f} (+/-{2:>8.4f})'.format(col, bestImpSummary.loc['mean', col], bestImpSummary.loc['std', col]))            
    
    print("=========================================================================")
    print("Cross Validation Score")
    print(finalScore)    
    print("=========================================================================")
    

# =============================================================================
#   Train the best Model by the worst validation set
# =============================================================================
    #trainWorst = pd.read_csv("tmp/trainXk_worst_f1_{0:.4f}.csv".format(min_acc), index_col=0)
    #validWorst = pd.read_csv("tmp/validXk_worst_f1_{0:.4f}.csv".format(min_acc), index_col=0)
    trainWorstIdx = pd.read_csv("tmp/trainXk_worst_set.csv", index_col=0, header=None)
    validWorstIdx = pd.read_csv("tmp/validXk_worst_set.csv", index_col=0, header=None)
     
    bestFeatureList = bestHyperParGroup[0]
    bestParList     = bestHyperParGroup[1]
    
    if modelType=='rf':
        
        bestModel = RandomForestClassifier(n_estimators=parList['Nestimators'],
                                           bootstrap=True,                                     
                                           criterion='gini',
                                           max_depth=parList['MaxDepth'],
                                           max_features='auto',
                                           #min_samples_split=,
                                           #min_samples_leaf=1,
                                           #min_weight_fraction_leaf=0.3,
                                           max_leaf_nodes=10, 
                                           oob_score=True,
                                           random_state=ranState,
                                           n_jobs=1,
                                           verbose=0)          
        
    elif modelType == 'xgb':           
        bestModel = XGBClassifier(n_estimators=parList['Nestimators'], 
                                  max_depth=parList['MaxDepth'], 
                                  eta=0.1,
                                  subsample=parList['subSample'])
        
        
    trainXX, validXX, trainYY, validYY = train_test_split(trainX[bestFeatureList], trainY, test_size=0.1, stratify=trainY)
    
    score_train, score_acc, score_f1  = just_fit(modelType, bestModel, 
                                                 #trainX=trainX[bestFeatureList].loc[trainWorstIdx.index], 
                                                 #trainY=trainY.loc[trainWorstIdx.index], 
                                                 #validX=trainX[bestFeatureList].loc[validWorstIdx.index], 
                                                 #validY=trainY.loc[validWorstIdx.index]
                                                 trainX=trainXX, 
                                                 trainY=trainYY, 
                                                 validX=validXX,                                                
                                                 validY=validYY
                                                 )
    print("Final Model")
    print("score_train: %.4f" %score_train)
    print("score_acc  : %.4f" % score_acc)
    print("score_f1   : %.4f" %score_f1)
    print("=========================================================================")
# =============================================================================
#   Prediction (Train + Test)
# =============================================================================
    
    predX = gen_train_data(allDF[trainDF.shape[0]:], onehotCate=onehotCate)            
    #print ('RF accuracy: TESTING',  modelRF.score(predX , y_test))      
    #predY = pd.DataFrame(index=testDF.index)
    #print(predX)
    
    #modelRF.fit(trainX[bestFeatureList], trainY)
    #model.fit(trainXk, trainYk,  early_stopping_rounds=100, eval_metric="error", eval_set=[(validXk, validYk)], verbose=False)
    
    result      = bestModel.predict(predX[bestFeatureList])
    resultTrain = bestModel.predict(trainX[bestFeatureList])
    
    predY      = pd.Series(result,      index=testDF.index,  name="Survived")
    predTrainY = pd.Series(resultTrain, index=trainDF.index, name="Survived")
    
    #outPath = 'titanic_out.csv'
    print("Dump: titanic_out.csv")                         
    predY.to_csv('titanic_out.csv', header=True)                             
    predTrainY.to_csv('titanic_train.csv', header=True)
    
# =============================================================================
#   Comapre with sample  
# =============================================================================
    print("*** Comapre with sample *** ")
    #sampleY = pd.read_csv("data/baseline_77511.csv", index_col=0)
    #sampleY = pd.read_csv("data/baseline_77033.csv", index_col=0)
    sampleY = pd.read_csv("data/baseline_79904.csv", index_col=0)
    #sampleY = pd.read_csv("data/baseline_73684.csv", index_col=0)
    
    #print(predY)
    #print(sampleY['Survived'])
    predY.name = 'Pred'        
    print(pd.crosstab(predY, sampleY['Survived']))
            
    compare = pd.concat([predY, sampleY['Survived'], testDF[['Sex', 'Pclass', 'Fare', 'Age']]], axis=1)
    
    #print(compare)
    #print(compare.drop(compare[['Pred']!= ['Survived']]))
    diff = compare[compare['Pred'] != compare['Survived']]
    #print(diff)
    diff.to_csv("diff.csv")
        
 