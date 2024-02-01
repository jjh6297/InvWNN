from __future__ import print_function
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Input, Dense, Permute, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
import os
import pickle
import h5py
import random
import cv2
import numpy as np
from resnet import ResNet18
from InvWNN import InvWNN, get_ConvLayer_pred, get_FCLayer_pred, get_BiasLayer_pred    
nb_classes = 10
NumLength=15


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
Percent = 0.5
ftr = sio.loadmat('Information_CIFAR10_ResNet18_LableNoise_3.mat')
Idxs = ftr['Idx'][0].tolist()
y_train_fake = ftr['y_train_fake']
y_train_fake = tf.keras.utils.to_categorical(y_train_fake, 10)
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

                
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip = True,
        shear_range=0.1,
        zoom_range=0.3,
        fill_mode='nearest',
        cval=0.,
)

def lr_schedule(epoch):
    lr = 1e-6
    return lr   
adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
lr_scheduler = LearningRateScheduler(lr_schedule)
batch_size = 256
callbacks = [lr_scheduler]

model = ResNet18()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])	
model.summary()
model.load_weights('Pretrained_ResNet_on_LabelNoise.h5')

model2_Conv = InvWNN(NumLength)
model2_Conv.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_Conv.load_weights('InvWNN_Conv.h5')
model2_FC = InvWNN(NumLength)
model2_FC.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_FC.load_weights('InvWNN_FC.h5')
model2_B = InvWNN(NumLength)
model2_B.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_B.load_weights('InvWNN_Bias.h5')


lcnt=0
WW = model.trainable_weights
WeightsSet=[]
for ww in range(len(WW)):
    if len(WW[ww].shape)>0:
        WeightsSet.append(np.zeros( WW[ww].shape + (NumLength,)))
        lcnt = lcnt+1  
        
        
Loss = []
Acc = []
ValLoss = []
ValAcc = []
cnt = 1

Selection = Idxs
Else = [x for x in list(range(50000)) if x not in Selection] 
print('Evaluation')
print(Selection)

output3 = model.evaluate(x_test,y_test, batch_size=128)
Target_Acc_Forget=[]
Test_Acc=[]
for loop in range(10):
    ii=0
    model.fit_generator(datagen.flow(x_train[Selection], y_train_fake[Selection], batch_size=batch_size),steps_per_epoch=len(Selection)//batch_size,epochs=2, verbose=1, workers=1, callbacks=[lr_scheduler])
    for _ in range(15):
        # Stacking Weight History
        lcnt=0
        WW = model.trainable_weights
        for ww in range(len(WW)):
            if len(WW[ww].shape)>0:
                WeightsSet[ww][...,ii]=np.array(WW[ww])
                lcnt = lcnt+1  
        model.fit_generator(datagen.flow(x_train[Selection], y_train_fake[Selection], batch_size=batch_size),steps_per_epoch=len(Selection)//batch_size,epochs=1, verbose=1, workers=1, callbacks=[lr_scheduler])
        ii=ii+1

    # Past Weight Prediction   
    print('Past Weight Prediction')    
    lcnt=0
    WW = model.trainable_weights
    for ww in range(len(WW)):
        if len(WW[ww].shape)==4 :
            NewWeight = get_ConvLayer_pred(WeightsSet[ww],model2_Conv,NumLength)
            WW[ww] = tf.cast(tf.constant(NewWeight), tf.float32)
            lcnt = lcnt+1    
        elif len(WW[ww].shape)==3 :
            NewWeight = get_DepthConvLayer_pred(WeightsSet[ww],model2_Conv,NumLength)
            WW[ww] = tf.cast(tf.constant(NewWeight), tf.float32)
            lcnt = lcnt+1     
        elif len(WW[ww].shape)==2 :
            NewWeight = get_FCLayer_pred(WeightsSet[ww],model2_FC,NumLength)
            WW[ww] = tf.cast(tf.constant(NewWeight), tf.float32)
            lcnt = lcnt+1    
        elif len(WW[ww].shape)==1 :
            NewBias = get_BiasLayer_pred(WeightsSet[ww],model2_B,NumLength)
            WW[ww] = tf.cast(tf.constant(NewBias), tf.float32)
            lcnt = lcnt+1    
    for ww in range(len(WW)):
        model.trainable_weights[ww].assign(WW[ww])
    print('Evaluation')

    output1 = model.evaluate(x_train[Selection], y_train_fake[Selection], batch_size=128)
    output3 = model.evaluate(x_test,y_test, batch_size=128)

    print(output3)
    Target_Acc_Forget.append(output1[1])
    Test_Acc.append(output3[1])


sio.savemat('Unlearning_History_ResNet_CIFAR10.mat', mdict = {'Target_Acc_Forget':np.array(Target_Acc_Forget),  'Test_Acc':np.array(Test_Acc) })

