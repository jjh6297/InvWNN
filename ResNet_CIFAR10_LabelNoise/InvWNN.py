from __future__ import print_function
import tensorflow as tf
import tensorflow.keras
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Lambda, Multiply, Add, Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

reg = 1e-6
reg2 = 0.1


def trAct_1D_Exp(x,compressed, numExp):

	xShape2=x.shape
	x2 = Dense(compressed, activation='relu')(x)
	temp = Dense(xShape2[1])(x2)
	for jj in range(2,numExp):
		temp = Add()([Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),Lambda(lambda x: K.exp(x))(Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),x]))]),temp])
	return temp	


    
def trAct_1D_Exp(x,compressed, numExp):

	xShape2=x.shape
	x2 = Dense(compressed, activation='relu')(x)
	temp = Dense(xShape2[1])(x2)
	for jj in range(2,numExp):
		temp = Add()([Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),Lambda(lambda x: K.exp(x))(Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),x]))]),temp])
	return temp	
    
    
def InvWNN(NumLength):
    inputs = Input(shape=(NumLength,))
    inputs2 = Input(shape=(NumLength-1,))

    fc = Dense(128, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'))(inputs)
    fc = LeakyReLU()(fc)
    
    fc = Dense(128, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'))(fc)
    fc = trAct_1D_Exp(fc,8,3)

    fc = Dense(64, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'))(fc)
    fc = LeakyReLU()(fc)

    fc2 = Dense(128, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'))(inputs2)
    fc2= LeakyReLU()(fc2)
    
    fc2 = Dense(128, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'))(fc2)
    fc2 = trAct_1D_Exp(fc2,8,3)

    fc2 = Dense(64, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'))(fc2)
    fc2= LeakyReLU()(fc2)

   
    fc = Dense(1,activation='tanh')(Concatenate()([fc,fc2]))

    network = Model([inputs,inputs2], fc)
    return network
    
    
def get_ConvLayer_pred(layer, predictor,NumLength):
    Layer = np.reshape(layer,[layer.shape[0]*layer.shape[1]*layer.shape[2]*layer.shape[3],layer.shape[4]])   
    Layer=Layer[:,::-1]    
    return np.reshape(Layer[:,NumLength-1] + predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0], [layer.shape[0], layer.shape[1], layer.shape[2], layer.shape[3]])



def get_DepthConvLayer_pred(layer, predictor, NumLength):
    Layer = np.reshape(layer,[layer.shape[0]*layer.shape[1]*layer.shape[2],layer.shape[3]]) 
    Layer=Layer[:,::-1]    
    diff =  predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0]   
    return np.reshape(Layer[:,NumLength-1] + diff, [layer.shape[0], layer.shape[1], layer.shape[2]]) 



def get_FCLayer_pred(layer, predictor, NumLength):
    Layer = np.reshape(layer,[layer.shape[0]*layer.shape[1], layer.shape[2]])   
    Layer=Layer[:,::-1]        
    return np.reshape(Layer[:,NumLength-1] + predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0], [layer.shape[0], layer.shape[1]])

    
    
def get_BiasLayer_pred(layer, predictor,NumLength):
    Layer=layer[:,::-1]    
    return Layer[:,NumLength-1] + predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0]
