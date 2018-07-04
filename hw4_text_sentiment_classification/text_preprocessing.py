# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 13:30:47 2018

@author: yilin9999
"""

import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer

class DocToken:
    def __init__(self, algo, maxWords, ngram_range=None):        
        self.algo        = algo
        self.maxWords    = maxWords
        self.ngram_range = ngram_range
        self.numWords    = 0
        self.embeddingMatrix = None
        
        if self.algo == 'bow':
            self.tok = TfidfVectorizer(max_features=maxWords, ngram_range=(1,ngram_range))
        else:
            self.tok = Tokenizer(num_words=maxWords)
            
    def fit(self, textX):
        if self.algo == 'bow':
            self.tok.fit(textX)
            self.numWords = min(self.maxWords, len(self.tok.vocabulary_))
        else:
            self.tok.fit_on_texts(textX)
            self.numWords = min(self.maxWords, len(self.tok.word_index)+1)
    def transform(self, textX):
        if self.algo == 'bow':
            return self.tok.transform(textX, 'tfidf').toarray()
        else:
            return self.tok.texts_to_sequences(textX)
            
    def pretrain_word_embeded(self, pretrain_wordvec_path):
        if self.algo == 'pretrain_word_embeded':
            self.embeddingMatrix = load_pretrained_wordvec(path=pretrain_wordvec_path,
                                                                wordNum=self.numWords,
                                                                wordIdxDict=self.tok.wordIndex, 
                                                                embeddingDim=100)
        else:
            raise ValueError("DocToken algorithm must be 'pretrain_word_embeded'")
            

def load_pretrained_wordvec(path, wordNum, wordIdxDict, embeddingDim):
    
    global predTrainVec
    
    predTrainVec = {}
    print("Loading %s" %path)        
    
    if len(predTrainVec) != 0:        
    
        with open(path, encoding='utf-8') as fptr:
            for line in fptr:
                token = line.split()
                word  = token[0]
                vec   = np.array(token[1:], dtype='float32')            
                predTrainVec[word] = vec
    
    
    EmbeddingMatrix = np.zeros((wordNum, embeddingDim))
    
    for word, idx in wordIdxDict.items():        
        if (word in predTrainVec) and (idx < wordNum) :
            EmbeddingMatrix[idx] = predTrainVec[word]
    #docToken
    
    return EmbeddingMatrix

def remove_stop_word(textX):
    
    #return textX
    nltk.download('punkt')    
    nltk.download('stopwords')
    
    new_textX = []
    stopWords = set(stopwords.words('english'))
    
    #print(stopWords)
    print("Start Removing stop words")
    for corpus in textX:
        str_list = []
        tmp = nltk.word_tokenize(corpus)        
        for tok in tmp:
            if tok not in stopWords:
                #tok.translate(None, string.punctuation)
                tok = re.sub(r'[^\w\s]','',tok)
                #tok = re.sub(r'a-zA-Z]','',tok)
                str_list.append(tok)
            
        new_textX.append(' '.join(str_list))
    
    #print(new_textX)
    return new_textX
