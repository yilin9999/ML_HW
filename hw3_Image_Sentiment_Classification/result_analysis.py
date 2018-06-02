# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 10:45:36 2018

@author: yilin9999
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix


def gen_confusion_matrix(model, classes, 
                         validX, validY,
                         paraDict, 
                         dumpfile=1, 
                         plot=1):
    
    predY       = model.predict_classes(validX, batch_size=paraDict['batchSize'], verbose=1)
    confuMatrix = confusion_matrix(validY, predY)
        
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()    
    print("TN={}, FP={}, FN={}, TP={}".format(tn, fp, fn, tp))
    
    if dumpfile==1:
        np.savetxt("confusion_matrix.csv", confuMatrix, delimiter=",", fmt='%d')
        print("dump confusion_matrix.csv")
    
    plot_confusion_matrix(cm=confuMatrix, classes=classes, plot=plot)


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet,
                          plot=1):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png',dpi=600)
    
    
    