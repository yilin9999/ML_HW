# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 08:44:08 2018

@author: yilin9999
"""
import dv_model
import dv_io
import dv_train

from dv_io import load_train_data

from argparse import ArgumentParser


myParser = ArgumentParser("text_setiment_classification")

myParser.add_argument('--train_data_path', 
                      type=str,
                      #default="C:\\testdata\\hw4_data\\training_label_small.txt", 
                      default="C:\\testdata\\hw4_data\\training_label.txt", 
                      dest='train_data_path',
                      help='train_data_path') 
    
myParser.add_argument('--epoch',        type=int,    default=3,    dest='epochNum',  help='epochNum')
myParser.add_argument('--batch_size',   type=int,    default=300,  dest='batchSize', help='batchSize')      
myParser.add_argument('--drop_rate' ,   type=float,  default=0,    dest='dropoutR' , help='dropoutR')     
myParser.add_argument('--neuro_cnt' ,   type=int,    default=512,  dest='neuroCnt' , help='neuroCnt')     
myParser.add_argument('--regular_rate', type=float,  default=0.01, dest='regularR' , help='regularR')   
myParser.add_argument('--plot      ',   type=bool,   default=False,dest='plot'  ,    help='plot')         
myParser.add_argument('--seed',         type=int,    default=0   , dest='seed' ,     help='seed')         

myArgs = myParser.parse_args()

paraDict = {'epochNum' : myArgs.epochNum,
            'batchSize': myArgs.batchSize,            
            'neuroCnt' : myArgs.neuroCnt,
            'regularR' : myArgs.regularR,
            'seed'     : myArgs.seed
            }

trainX, trainY = load_train_data(myArgs.train_data_path)
paraDict['vecSize'] = trainX.shape[1]

myModel = dv_model.create_dnn(paraDict)

dv_train.train(trainX, trainY, myModel, paraDict)


