# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential  
from keras.layers import Dense  
import matplotlib.pyplot as plt  


def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  


(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()  

print("\t[Info] train data={:7,}".format(len(X_train_image)))  
print("\t[Info] test  data={:7,}".format(len(X_test_image)))  

print("\t[Info] Shape of train data=%s" % (str(X_train_image.shape)))  
print("\t[Info] Shape of train label=%s" % (str(y_train_label.shape)))  

  
model = Sequential()  # Build Linear Model  
  
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) # Add Input/hidden layer  
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax')) # Add Hidden/output layer  
print("\t[Info] Model summary:")  
model.summary()  
print("")  

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)  
show_train_history(train_history, 'acc', 'val_acc') 

scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  


#print(list(''.join(myList)))
#return combinations(myStr, len(str(digits)))