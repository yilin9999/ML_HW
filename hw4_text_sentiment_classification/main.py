# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:44:08 2018

@author: yilin9999
"""
import dv_model

#from dv_io import InputData
from dv_io import DataManerger
from dv_io import dump_pkl 
from dv_analysis import gen_confusion_matrix

import text_preprocessing as textp
from text_preprocessing import DocToken
import time
from argparse import ArgumentParser

import os

    
myParser = ArgumentParser("text_setiment_classification")

myParser.add_argument('--train_data_path', 
                      type=str,
                      default="C:\\testdata\\hw4_data\\training_label_small.txt", 
                      #default="C:\\testdata\\hw4_data\\training_label.txt", 
                      dest='train_data_path',
                      help='train_data_path') 

myParser.add_argument('--semi_data_path', 
                      type=str,
                      default="C:\\testdata\\hw4_data\\training_nolabel_small.txt",                       
                      dest='semi_data_path',
                      help='semi_data_path')     

myParser.add_argument('--pred_data_path', 
                      type=str,
                      default="C:\\testdata\\hw4_data\\testing_data_small.txt",                       
                      #default=None,                       
                      dest='pred_data_path',
                      help='pred_data_path')     

myParser.add_argument('--pretrain_wordvec_path', 
                      type=str,
                      #default=None,                       
                      default="C:\\testdata\\hw4_data\\glove.6B.100d.txt",                       
                      dest='pretrain_wordvec_path',
                      help='pretrain_wordvec_path')     



#myParser.add_argument('-word_embeded'    , action='store_true', dest='wordEmbeded', help='wordEmbeded')              
myParser.add_argument('-remove_stopwords', action='store_true', dest='rmStopwords', help='rmStopwords')              

#myParser.add_argument('-word_embeded'        , action='store_false', dest='wordEmbeded', help='wordEmbeded')        
#myParser.add_argument('-remove_stopwords', action='store_false', dest='rmStopwords', help='rmStopwords')              



myParser.add_argument('--epoch',        type=int,    default=3,     dest='epochNum',  help='epochNum')
myParser.add_argument('--batch_size',   type=int,    default=2,    dest='batchSize', help='batchSize')
myParser.add_argument('--max_vob_size', type=int,    default=500,  dest='maxVocabSize' ,help='maxVocabSize')               
myParser.add_argument('--max_seq_len' , type=int,    default=30,  dest='maxSeqLen' ,help='maxSeqLen')               
myParser.add_argument('--algo       ' , type=str,    default='bow',  dest='algo' ,help='algo', choices=['bow', 'auto_word_embeded','pretrain_word_embeded'])
myParser.add_argument('--ngram_bow',     type=int,   default=1,  dest='ngramBow' ,help='ngramBow', choices=range(1,5))
myParser.add_argument('--drop_rate' ,   type=float,  default=0,    dest='dropoutR' , help='dropoutR')     
myParser.add_argument('--dnn_layers' ,  type=int,    default=2,    dest='dnnLayers' ,help='dnnLayers')     
myParser.add_argument('--neuro_cnt' ,   type=int,    default=512,  dest='neuroCnt' , help='neuroCnt')     
myParser.add_argument('--regular_rate', type=float,  default=0.01, dest='regularR' , help='regularR')   
myParser.add_argument('--plot      ',   type=bool,   default=False,dest='plot'  ,    help='plot')         
myParser.add_argument('--seed',         type=int,    default=0   , dest='seed' ,     help='seed')         
myParser.add_argument('--semi_round',   type=int,    default=2   , dest='semiRound' ,help='semiRound')         
myParser.add_argument('--semi_thd',   type=float,    default=0.6 , dest='semiThd' ,help='semiThd')         
myParser.add_argument('--max_data_size',   type=int, default=100 , dest='maxDataSize' ,help='maxDataSize')         

myArgs = myParser.parse_args()

semiTrain       = myArgs.semiRound != 0
#usePretrianWord = (myArgs.pretrain_wordvec_path != None)

paraDict = {'epochNum' : myArgs.epochNum,            
            'algo'     : myArgs.algo,
            'ngramBow' : myArgs.ngramBow,
            'maxVocabSize':myArgs.maxVocabSize,
            'maxSeqLen': myArgs.maxSeqLen,
            'batchSize': myArgs.batchSize,     
            'dropoutR' : myArgs.dropoutR,
            'dnnLayers': myArgs.dnnLayers,
            'neuroCnt' : myArgs.neuroCnt,
            'regularR' : myArgs.regularR,
            'seed'     : myArgs.seed,
            'semiTrain': semiTrain,
            'rmStopwords':myArgs.rmStopwords,
            'startTime' : time.strftime("%m-%d-%H-%M", time.localtime())            
            }


    
################################################################################
## Load Data
################################################################################    
#trainData = InputData('train')
myData = DataManerger()
myData.append_new_data(myArgs.train_data_path, 'train')


if semiTrain==True:    
    myData.append_new_data(myArgs.semi_data_path, 'semi')    


if len(myData.trainX) > myArgs.maxDataSize:
    myData.trainX = myData.trainX[:myArgs.maxDataSize]
    myData.trainY = myData.trainY[:myArgs.maxDataSize]

if len(myData.semiX) > myArgs.maxDataSize:
    myData.semiX = myData.semiX[:myArgs.maxDataSize]
    myData.semiY = myData.semiY[:myArgs.maxDataSize]


if paraDict['rmStopwords'] is True:
    myData.trainX = textp.remove_stop_word(myData.trainX)
    myData.semiX = textp.remove_stop_word(myData.semiX)

#print(myData.trainX)
#sys.exit(0)

myData.split_valid_from_train(0.1, shuffle=True)    


"""
if paraDict['algo']=='bow':
    docToken = TfidfVectorizer(max_features=paraDict['maxVocabSize'], ngram_range=(1,paraDict['ngramBow']))
    docToken.fit(itertools.chain(myData.trainX, myData.semiX))
    paraDict['maxVocabSize'] = min(paraDict['maxVocabSize'], len(docToken.vocabulary_))
    #docToken.max_features = paraDict['maxVocabSize'] 
    
    print("Orignal word number: %d" %(paraDict['maxVocabSize']))
    print("Train word number  : %d" %(len(docToken.vocabulary_)))
else:
    ##Kears Method
    docToken = Tokenizer(num_words=paraDict['maxVocabSize'])
    #docToken = Tokenizer(num_words=paraDict['maxVocabSize'], split=' ', filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
    docToken.fit_on_texts(itertools.chain(myData.trainX, myData.semiX))

    wordIndex = docToken.word_index
    paraDict['maxVocabSize'] = min(paraDict['maxVocabSize'], len(wordIndex)+1)
    docToken.num_words= paraDict['maxVocabSize']
    #docToken.fit_on_texts(itertools.chain(myData.trainX, myData.semiX))
    #print(docToken.word_index)
    
    if paraDict['algo']=='pretrain_word_embeded':        
        EmbeddingMatrix = textp.load_pretrained_wordvec(path=myArgs.pretrain_wordvec_path,
                                                            wordNum=paraDict['maxVocabSize'],
                                                            wordIdxDict=wordIndex, 
                                                            embeddingDim=100)
    else:
        EmbeddingMatrix = None
"""


################################################################################    
## Train
################################################################################
    
print(paraDict)


#if semiTrain==True:
for i in range(myArgs.semiRound + 1):   
    
    paraDict['outDir'] = "%s_round_%d" %(paraDict['startTime'], i)     
    if not os.path.exists(paraDict['outDir']):
        os.makedirs(paraDict['outDir'])
        
    print("########## Trainning Round %d ##########" % i)
         
    ################################################################################
    ## Word Embedding
    ################################################################################
    print("Start word embedding")
    docToken = DocToken(paraDict['algo'], paraDict['maxVocabSize'], ngram_range=paraDict['ngramBow'])
    docToken.fit(myData.trainX)   
    
    print("Orignal word number: %d" %paraDict['maxVocabSize'])
    print("Train word number  : %d" %docToken.numWords)
    paraDict['maxVocabSize'] = docToken.numWords    


    
    ################################################################################
    ## Build Model 
    ################################################################################
    
    if paraDict['algo'] == 'bow':
        myModel = dv_model.create_dnn(paraDict)
    else:
        myModel = dv_model.create_lstm_dnn(paraDict, docToken.embeddingMatrix)        
    
    dv_model.train(myModel, paraDict, docToken, 
                   myData.trainX, 
                   myData.trainY, 
                   myData.validX, 
                   myData.validY)

    ################################################################################
    ## Output
    ################################################################################               
    dump_pkl(docToken, '%s/word_embeded.pkl' %paraDict['outDir'])
    dump_pkl(paraDict, '%s/paraDict.pkl'     %paraDict['outDir'])

    #semiX = myData.get_semi_dataX()           
    
    ################################################################################
    ## Semi-supervised
    ################################################################################          
    if myArgs.semiRound>0 and len(myData.semiX) != 0:
        predY = dv_model.predict(myModel, paraDict, docToken, myData.semiX)
        #myData.set_semi_dataY(semiY)        
        myData.psudo_lable(predY, thd=myArgs.semiThd)                
        paraDict['regularR'] *= 2
        """
        dv_model.train(myModel, paraDict, docToken,
                       myData.trainX,   
                       myData.trainY, 
                       myData.validX, 
                       myData.validY)
        """
    #end for loop
        
print("Generate Confusion Matrix")
predY = dv_model.predict(myModel, paraDict, docToken, myData.validX, pred_class=True)    
gen_confusion_matrix(myData.validY, predY, ['Negative','Positive'], dumpfile="%s/confusion_matrix"%paraDict['outDir'])

#myModel.save("model_%s.h5", paraDict['startTime'])
#semiY = myModel.predict()

########## Prediction ##########
if myArgs.pred_data_path is not None:    
    myData.append_new_data(myArgs.pred_data_path, 'test')    
    dv_model.predict(myModel, paraDict, docToken, myData.testX)
    


