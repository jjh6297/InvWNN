import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Permute, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar100
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import tfimm
import gc
import time
import datetime
import pickle
import copy

NumLength=15
reg = 1e-6
reg2 = 0.1
global model2_Conv
global model2_FC
global model2_B
global prediction
global NewWeight
global NewBias


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
    
    

def get_ConvLayer_pred(layer, model2_Conv):
    print(layer.shape)
    Layer = np.reshape(layer,(-1,layer.shape[-1]))    
    Layer=Layer[:,::-1]   
    diff = model2_Conv.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0]    
    global NewWeight; NewWeight=np.reshape(Layer[:,NumLength-1] + diff, layer.shape[:-1])   
    return NewWeight        

def get_DepthConvLayer_pred(layer, model2_Conv):
    print(layer.shape)
    Layer = np.reshape(layer,(-1,layer.shape[-1]))    
    Layer=Layer[:,::-1]           
    diff =  model2_Conv.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0]   
    global NewWeight; NewWeight= np.reshape(Layer[:,NumLength-1] + diff, layer.shape[:-1]) 
    return NewWeight        

def get_FCLayer_pred(layer, model2_FC):
    Layer = np.reshape(layer,(-1,layer.shape[-1]))  
    Layer=Layer[:,::-1]           
    diff = model2_FC.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0] 
    global NewWeight; NewWeight= np.reshape(Layer[:,NumLength-1] + diff, layer.shape[:-1])   
    return NewWeight        

def get_BiasLayer_pred(layer, model2_B):
    Layer=layer[:,::-1]    
    diff = model2_B.predict([layer, layer[:,1:NumLength] - layer[:,0:NumLength-1]], batch_size = 100000)[:,0]   
    global NewBias; NewBias= Layer[:,NumLength-1] + diff	
    return NewBias        

        
        
model2_Conv = InvWNN(NumLength)
model2_Conv.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_Conv.load_weights('InvWNN_Conv.h5')
model2_FC = InvWNN(NumLength)
model2_FC.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_FC.load_weights('InvWNN_FC.h5')
model2_B = InvWNN(NumLength)
model2_B.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_B.load_weights('InvWNN_Bias.h5')
    
    


@tf.function
def image_augmentation(images,
                       width_shift_range=0., height_shift_range=0.,
                       rotation_range = 0.0, 
                       horizontal_flip=False,
                       vertical_flip=False,
                       cval=0.0, cutout_size=None, cutout_num=1, 
                       random_color_p=0.0):

    img_shape = images.shape[-3:]
    img_width = img_shape[1]
    img_height = img_shape[0]
    interpolation  = 'BILINEAR' if rotation_range!=0.0 else 'NEAREST'

    def transform(image):
        mirror_x = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.dtypes.int32)*2-1, tf.float32) if horizontal_flip else 1.0
        mirror_y = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.dtypes.int32)*2-1, tf.float32) if vertical_flip else 1.0
        width_shift = tf.random.uniform(shape=[], minval=-width_shift_range, maxval=width_shift_range)*img_width
        height_shift = tf.random.uniform(shape=[], minval=-height_shift_range, maxval=height_shift_range)*img_width
        zoom_x = 1.0
        zoom_y = 1.0
        center = img_width/2

        angle = tf.random.uniform(shape=[], minval=-rotation_range, maxval=rotation_range)*3.141519/180
        sinval = tf.sin(angle)
        cosval = tf.cos(angle)
        center_mat = [1.0, 0.0, center, 0.0, 1.0, center, 0.0, 0.0]
        rotate_mat = [cosval, -sinval, 0.0, sinval, cosval, 0.0, 0.0, 0.0]
        zoom_mat = [zoom_x*mirror_x, 0.0, 0.0, 0.0, zoom_y*mirror_y, 0.0, 0.0, 0.0]
        center_mat_inv = [1.0, 0.0, width_shift-center, 0.0, 1.0, height_shift-center, 0.0, 0.0]
        matrix = [center_mat, rotate_mat, zoom_mat, center_mat_inv]
        composed_matrix = tfa.image.transform_ops.compose_transforms(matrix)
        (h, w, c) = (img_shape[0], img_shape[1], img_shape[2])
        images = tf.reshape( image, [1, h, w, c] )
        images = tf.raw_ops.ImageProjectiveTransformV2(
            images=images, transforms=composed_matrix, output_shape=[h, w], 
            fill_mode='REFLECT', interpolation=interpolation)
        image = tf.reshape( images, [h, w, c] )
        return image

    def cutout(image, cval=0, cnum = 1, csize = 0.25):
        DIM = image.shape[0]
        for k in range( cnum ):
            x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
            y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

            WIDTH = tf.cast( csize*DIM,tf.int32)
            ya = tf.math.maximum(0,y-WIDTH//2)
            yb = tf.math.minimum(DIM,y+WIDTH//2)
            xa = tf.math.maximum(0,x-WIDTH//2)
            xb = tf.math.minimum(DIM,x+WIDTH//2)

            one = image[ya:yb,0:xa,:]
            two = tf.fill([yb-ya,xb-xa,3], tf.cast(cval, image.dtype) ) 
            three = image[ya:yb,xb:DIM,:]
            middle = tf.concat([one,two,three],axis=1)
            image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM,:,:]],axis=0)

        image = tf.reshape(image,[DIM,DIM,3])
        return image
    
    def random_color(image, prob):
        img_tmp = tf.image.random_contrast(image, 0.8, 1.2)
        img_tmp = tf.image.random_saturation(img_tmp, 0.5, 1.5)
        p = tf.random.uniform([],0.0,1.0)
        return tf.where(p<prob, tf.cast(img_tmp, image.dtype) , image)

    images = tf.map_fn(lambda image: transform(image), images)

    if cutout_size!=None and cutout_num!=0:
        images = tf.map_fn(lambda image: cutout(image,cval=cval, csize=cutout_size, cnum=cutout_num), images)

    if random_color_p!=0.0:
        images = tf.map_fn( lambda image: random_color(image, prob=random_color_p), images)

    return images




class TransferTrainer:

    def make_dataset( self,train_data, validation_data, batch_size):
        (x_train, label_train)= train_data
        (x_test, label_test)=validation_data
        
        train_len = len(x_train)
        test_len = len(x_test)

        ds_train = tf.data.Dataset.from_tensor_slices(train_data)
        ds_train = ds_train.shuffle(train_len).batch(batch_size,drop_remainder=True)
        ds_validation = tf.data.Dataset.from_tensor_slices(validation_data)
        ds_validation = ds_validation.batch(batch_size)

        self.batch_size=batch_size
        self.ds_train = ds_train
        self.ds_validation = ds_validation

    def build_model( self, input_shape=(32,32,3),  num_classes=10, dropout=0.25,resolution=1.0):
        original_input_size=224
        preprocess = tfimm.create_preprocessing("pvt_v2_b0", dtype="float32")
        original_input_size = round(resolution*original_input_size)//8*8
        base_input_shape = (original_input_size,original_input_size,3)
        self.base_model = tfimm.create_model(model_name= "pvt_v2_b0", nb_classes=0)
        
        try:
            self.base_model.load_weights('pvt_v2_b0_pretrained.h5', by_name=True, skip_mismatch=True)
        except:
            self.base_model_o = tfimm.create_model("pvt_v2_b0", pretrained="[pytorch]https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth")#tfimm.create_model(model_name= "vit_tiny_patch16_224", pretrained="timm",nb_classes=0)
            self.base_model_o.save_weights('pvt_v2_b0_pretrained.h5')        
            self.base_model.load_weights('pvt_v2_b0_pretrained.h5', by_name=True, skip_mismatch=True)
        
        x = inputs = tf.keras.layers.Input(shape=input_shape)
        if input_shape != base_input_shape:
            x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, base_input_shape[0:2]), 
                                       output_shape=base_input_shape, name='stem_resize')(x)
        x = self.base_model(x, training=False)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(num_classes, name='prediction')(x)
        outputs = tf.keras.layers.Activation('softmax')(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.preprocess = preprocess

    def _get_augmented_dateset(self, dataset, augment=True, preprocess=None, cutout_num=0, cutout_size=0.25, shift_range=0.0, rotation_range=0.0, random_color= 0.0):
           
        if augment:
            def data_augmentation(image,label):
                image = image_augmentation(image, shift_range, shift_range, rotation_range, 
                                           horizontal_flip=True, cval=127.0, cutout_size=cutout_size, cutout_num=cutout_num,
                                           random_color_p=random_color )
                return image, label
            dataset = dataset.map(lambda image, label: data_augmentation(image, label ))

        if preprocess != None:
            def data_preprocesssing(image,label):
                image = tf.cast(image, tf.float32)
                image = preprocess(image)
                return image, label
            dataset = dataset.map( lambda image, label: data_preprocesssing(image, label) )

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def train(self,burn_in_epochs, finetuning_epochs, opt_class, opt_kwargs, lr_scheduler,ckp, burnin_lr=1e-1, 
              freeze_ratio=0.0, cutout_num=1, cutout_size=0.25, shift_range=0.2, random_color=1.0, label_smoothing=0.0, steps_per_execution=None):
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        acc = tf.keras.metrics.CategoricalAccuracy(name='acc')
        
        compile_kwargs = { 'loss':loss,'metrics':[acc]}
        global prediction
        global NewWeight
        global NewBias
 

        lcnt=0
        WW = self.model.trainable_weights
        WeightsSet=[]
        for ww in range(len(WW)):
            if len(WW[ww].shape)>0:
                WeightsSet.append(np.zeros( WW[ww].shape + (NumLength,)))
                lcnt = lcnt+1      
        optimizer = opt_class(**opt_kwargs)
        self.model.compile(optimizer=optimizer,  **compile_kwargs  )
        self.model.summary()

        ds_train      = self._get_augmented_dateset(self.ds_train, True, self.preprocess, cutout_num, cutout_size, shift_range, random_color=random_color)
        ds_train2      = self._get_augmented_dateset(self.ds_train, False, self.preprocess)
        ds_validation = self._get_augmented_dateset(self.ds_validation, False, self.preprocess)

        Target_Acc=[]
        Test_Acc=[]  
        output1 = self.model.evaluate(ds_train2)
        output3 = self.model.evaluate(ds_validation)

        Target_Acc.append(output1[1])
        Test_Acc.append(output3[1])

 
        for loop in range(30):
            ii=0
            for _ in range(NumLength):
                lcnt=0
                WW = self.model.trainable_weights
                for ww in range(len(WW)):
                    if len(WW[ww].shape)>0:
                        WeightsSet[ww][...,ii]=np.array(WW[ww])
                        lcnt = lcnt+1      
                print(lcnt)  

                self.model.fit(ds_train, epochs=1, callbacks=[lr_scheduler])

                ii=ii+1

            lcnt=0
            WW = self.model.trainable_weights
            for ww in range(len(WW)):
                if len(WW[ww].shape)==4 :
                    NewWeight = get_ConvLayer_pred(WeightsSet[ww],model2_Conv)
                    WW[ww] = tf.cast(tf.constant(NewWeight), tf.float32)
                    lcnt = lcnt+1    
                elif len(WW[ww].shape)==3 :
                    NewWeight = get_DepthConvLayer_pred(WeightsSet[ww],model2_Conv)
                    WW[ww] = tf.cast(tf.constant(NewWeight), tf.float32)
                    lcnt = lcnt+1     
                elif len(WW[ww].shape)==2 :
                    NewWeight = get_FCLayer_pred(WeightsSet[ww],model2_FC)
                    WW[ww] = tf.cast(tf.constant(NewWeight), tf.float32)
                    lcnt = lcnt+1    
                elif len(WW[ww].shape)==1 :
                    NewBias = get_BiasLayer_pred(WeightsSet[ww],model2_B)
                    WW[ww] = tf.cast(tf.constant(NewBias), tf.float32)
                    lcnt = lcnt+1    
 
            for ww in range(len(WW)):
                self.model.trainable_weights[ww].assign(WW[ww])
            print('Evaluation')
            output1 = self.model.evaluate(ds_train2)
            output3 = self.model.evaluate(ds_validation)
            print(output1)
            print(output3)
            Target_Acc.append(output1[1])
            Test_Acc.append(output3[1])
            del NewWeight
            del NewBias
            gc.collect()
        sio.savemat('Unlearning_History_PVTV2_CIFAR100.mat', mdict = {'Target_Acc':np.array(Target_Acc), 'Test_Acc':np.array(Test_Acc)})

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

Percent = 0.2
ftr = sio.loadmat('Information_CIFAR100_PVTV2_2_LabelNoise.mat')
Idxs = ftr['Idx'][0].tolist()
y_train_fake = ftr['y_train_fake']

Selection = Idxs
Else = [x for x in list(range(50000)) if x not in Selection] 

x_train=x_train[Idxs]
y_train=y_train_fake[Idxs]

trial=2




y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)
train_data, validation_data = (x_train,y_train), (x_test,y_test)




model_dict ={
    'ResNet50V2' : (tf.keras.applications.ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input, 224), 
    'ResNet101V2' : (tf.keras.applications.ResNet101V2, tf.keras.applications.resnet_v2.preprocess_input, 224), 
    'ResNet152V2' : (tf.keras.applications.ResNet152V2, tf.keras.applications.resnet_v2.preprocess_input, 224), 
    'Xception' : (tf.keras.applications.Xception, tf.keras.applications.xception.preprocess_input, 299),
    'EfficientNetB0' : (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input, 224),
    'EfficientNetB1' : (tf.keras.applications.EfficientNetB1, tf.keras.applications.efficientnet.preprocess_input, 240),
    'EfficientNetB2' : (tf.keras.applications.EfficientNetB2, tf.keras.applications.efficientnet.preprocess_input, 260),
    'EfficientNetB3' : (tf.keras.applications.EfficientNetB3, tf.keras.applications.efficientnet.preprocess_input, 300),
    'EfficientNetB4' : (tf.keras.applications.EfficientNetB4, tf.keras.applications.efficientnet.preprocess_input, 380),
    'EfficientNetB5' : (tf.keras.applications.EfficientNetB5, tf.keras.applications.efficientnet.preprocess_input, 456),
    'EfficientNetB6' : (tf.keras.applications.EfficientNetB6, tf.keras.applications.efficientnet.preprocess_input, 528),
    'EfficientNetB7' : (tf.keras.applications.EfficientNetB7, tf.keras.applications.efficientnet.preprocess_input, 600),
    'NASNetMobile' : (tf.keras.applications.NASNetMobile, tf.keras.applications.nasnet.preprocess_input, 224),
    'InceptionResNet' : (tf.keras.applications.InceptionResNetV2, tf.keras.applications.inception_resnet_v2.preprocess_input, 299),
}

def train(use_tpu=False):
        freeze_ratio=0.0
        burnin_epochs = 5


        optimizer= tf.keras.optimizers.SGD
        opt_kwargs = {'momentum':0.9}
        batch_size = 128
        burnin_lr = 0.01
        cutout_num = 2
        cutout_size = 0.4
        random_color = 1.0
        barnin_epochs = 15
        warmup_epochs = 15
        flat_epochs = 15
        cooldown_epochs = 60
        min_lr  = 0.0001
        max_lr = 0.0025
        dropout = 0.
        label_smoothing = 0.0
        resolution = 1.0

        model_name = 'pvt_v2_b0'

        trainer = TransferTrainer()
        print('make_data')
        trainer.make_dataset(train_data,validation_data, batch_size)

        total_epochs = warmup_epochs+flat_epochs+cooldown_epochs
        message = f'{model_name} {optimizer.__name__} epochs={burnin_epochs}+{warmup_epochs}+{flat_epochs}+{cooldown_epochs} batch_size={batch_size} lr={burnin_lr}/{max_lr}/{min_lr}'
        message += f' cutout={cutout_size}x{cutout_num} resolution={resolution} random_color={random_color} dropout={dropout} label_smoothing={label_smoothing}'

        print('start', message)
        start_time = time.time()
        def scheduler(epoch, lr):
            return 1e-5 

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        ckp = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./Checkpoint', 'Clean_'+model_name+'_weights.{epoch:02d}-{val_acc:.2f}.hdf5'),monitor='val_acc',save_weights_only=True,verbose=1)

        trainer.build_model( input_shape=(32,32,3), num_classes=100, 
                            resolution=resolution, dropout=dropout)
        trainer.model.layers[2].summary()                    
        trainer.model.load_weights('Pretrained_PVTv2_on_LabelNoise.h5')

        trainer.train( barnin_epochs, total_epochs, optimizer, opt_kwargs, 
                        lr_scheduler,ckp, burnin_lr, freeze_ratio=freeze_ratio, 
                        cutout_num=cutout_num, cutout_size=cutout_size, random_color=random_color,
                        label_smoothing=label_smoothing, steps_per_execution=10 if use_tpu else 1)
        K.clear_session()
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu) 
except ValueError:
    tpu=None
    strategy = tf.distribute.get_strategy()


with strategy.scope():
    train(tpu!=None)        