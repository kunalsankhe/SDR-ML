#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:59:50 2017

@author: shamnaz
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:51:57 2017

@author: shamnaz
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 16:51:46 2017

@author: shamnaz
"""

# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
#%matplotlib inline
import os,random
#os.environ["KERAS_BACKEND"] = "theano"
os.environ["KERAS_BACKEND"] = "tensorflow"
#os.environ["THEANO_FLAGS"]  = "device=cuda%d"%(6)
import numpy as np
#import theano as th
#import tensorflow as T
from sklearn.metrics import roc_auc_score, roc_curve, auc
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import adam
from keras.regularizers import l2
#from keras import regularizers
#import matplotlib.pyplot as plt
#import seaborn as sns
import cPickle, random, sys, keras
import h5py
from keras import backend as K
K.set_image_dim_ordering('th')
from scipy.io import loadmat
import pickle
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
#RESHAPING INPUT
#x=y.reshape(15640,2,128)

dist='14'
n_classes = 16
file1='IQ'+str(n_classes)+'_'+ dist +'ft_train_'+str(n_classes*200)+'K.mat'
f = h5py.File(file1,'r')
X = f.get('MatrixData')
X = np.array(X)
X = X.transpose()
X_train = X.astype('float32')

del X  

l2_lambda = 0.0001
seg=128

X_trainsize = (X_train.shape[0]/(seg))
print X_trainsize
L=X_train[:,0].reshape(X_trainsize,1,seg)
M=X_train[:,1].reshape(X_trainsize,1,seg)

meanL=np.mean(L)
meanM=np.mean(M)
std_L=np.std(L)
std_M=np.std(M)

L=L-meanL
M=M-meanM

L=L/std_L
M=M/std_M

X_train=np.concatenate([L, M], axis=1)

del L, M



y_trainsize = X_trainsize/n_classes
y_train=[]
for idx in range(0,n_classes):
    y_train=np.concatenate((y_train, idx*np.ones(y_trainsize,)), axis=0 )

y_train = np_utils.to_categorical(y_train,n_classes)



file2='IQ'+str(n_classes)+'_'+ dist +'ft_test_'+str(n_classes*50)+'K.mat'
f = h5py.File(file2,'r')
X = f.get('MatrixData')
X = np.array(X)    
X = X.transpose()
X_test = X.astype('float32')


del X 

#matFile = 'Disp128_test_200K.mat'
#mat = loadmat(matFile)
#X_test= mat['MatrixData']
X_testsize = (X_test.shape[0]/(seg))

print X_testsize

L=X_test[:,0].reshape(X_testsize,1,seg)
M=X_test[:,1].reshape(X_testsize,1,seg)

L=L-meanL
M=M-meanM

L=L/std_L
M=M/std_M

X_test=np.concatenate([L, M], axis=1)

del L, M

y_testsize = X_testsize/n_classes

y_test=[]
for idx in range(0,n_classes):
    y_test=np.concatenate((y_test, idx*np.ones(y_testsize,)), axis=0 )

#y1 = np.zeros(y_testsize,)
#y2 = np.ones(y_testsize,)
#y3 = 2*np.ones(y_testsize,)
#y4 = 3*np.ones(y_testsize,)
#y5 = 4*np.ones(y_testsize,)
#y_test = np.concatenate((y1,y2, y3, y4,y5), axis=0)
y_test = np_utils.to_categorical(y_test,n_classes)

print len(y_test)
#del y1, y2, y3, y4,y5

#test_idx = np.random.choice(range(0,X_testsize), size=X_testsize, replace=False)
#
#X_test=X_test[test_idx]
#y_test=y_test[test_idx]
file3='IQ'+str(n_classes)+'_'+ dist +'ft_validation_'+str(n_classes*10)+'K.mat'
f = h5py.File(file3,'r')
X = f.get('MatrixData')
X = np.array(X)
X = X.transpose()
X_val = X.astype('float32')

del X  


X_valsize = (X_val.shape[0]/(seg))
print X_val
L=X_val[:,0].reshape(X_valsize,1,seg)
M=X_val[:,1].reshape(X_valsize,1,seg)

L=L-meanL
M=M-meanM

L=L/std_L
M=M/std_M

X_val=np.concatenate([L, M], axis=1)

del L, M



y_valsize = X_valsize/n_classes
y_val=[]
for idx in range(0,n_classes):
    y_val=np.concatenate((y_val, idx*np.ones(y_valsize,)), axis=0 )

y_val = np_utils.to_categorical(y_val,n_classes)



k_scores = []
roc_all = []
fpr_all = []
tpr_all = []
overacc_all = []


in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
   
dr = 0.5 # dropout rate (%)
flt1=50
flt2=50
taps=7
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(flt1, 1, taps, border_mode='valid', name="conv1",kernel_regularizer = l2(l2_lambda), init='glorot_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(flt2, 2, taps, border_mode="valid",  name="conv2", kernel_regularizer = l2(l2_lambda),init='glorot_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,  init='he_normal',kernel_regularizer = l2(l2_lambda), name="dense1"))
model.add(Activation('relu'))
model.add(Dropout(dr))
model.add(Dense(80, init='he_normal',kernel_regularizer=l2(l2_lambda), name="dense2"))
model.add(Activation('relu'))
model.add(Dropout(dr))
model.add(Dense( n_classes, init='he_normal',kernel_regularizer = l2(l2_lambda), name="dense3" ))
model.add(Activation('softmax'))
model.add(Reshape([n_classes]))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

# Set up some params 
nb_epoch = 100     # number of epochs to train on
batch_size = 1024  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'IQ_dense_'+str(n_classes)+'_'+dist+'ft.h5'
history = model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=2,
                    shuffle=True,
                    validation_data=(X_val, y_val),
                    callbacks = [
                            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                            ])
#model.save('disp_small_slide1_8M_model.h5')
# we re-load the best weights once training is finished
#Assuming you have code for instantiating your model, 
#you can then load the weights you saved into a model with the same architecture:
model.save(filepath)


# Show simple version of performance
score = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)
print score
k_scores.append(score)



# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([n_classes,n_classes])
confnorm = np.zeros([n_classes,n_classes])
for i in range(0,X_test.shape[0]):
    j = list(y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
print "confusion matrix"    
print conf
for i in range(0,n_classes):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
print "confnormal"    
print confnorm
cor = np.sum(np.diag(conf))
ncor = np.sum(conf) - cor
overacc = cor / (cor+ncor)
print "Overall Accuracy: ", overacc 
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], test_Y_hat[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

roc_all.append(roc_auc)
fpr_all.append(fpr)
tpr_all.append(tpr)
overacc_all.append(overacc)  
        
meanacc = np.mean(overacc_all)

file_pickle='IQ_dense_'+str(n_classes)+'_'+dist+'ft.pickle'
with open(file_pickle, 'w') as f:
    # Python 3: open(..., 'wb')
    pickle.dump([roc_all, fpr_all, tpr_all, overacc_all, meanacc], f)




