#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import cv2
import h5py

import tensorflow as tf

import tensorflow_addons as tfa
import tensorflow.keras.layers as layer

import os

print("Imports complete")


# In[5]:


#!pip install opencv-python
#!pip install tensorflow


# In[31]:





# In[42]:


dir = r"tmp\train_data"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir, labels = "inferred", label_mode = "int", class_names = ['COVID','Viral Pneumonia'],
    color_mode = "rgb", batch_size = 32, image_size = (224, 224), 
    shuffle = True, seed = 42, validation_split = 0.1, subset = "training", interpolation = "bicubic")

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir, labels = "inferred", label_mode = "int", class_names = ['COVID','Viral Pneumonia'],
    color_mode = "rgb", batch_size = 32, image_size = (224, 224), 
    shuffle = True, seed = 42, validation_split = 0.1, subset = "validation", interpolation = "bicubic")


# In[43]:


class_names = train_ds.class_names
print(class_names)


# In[44]:


train_ds = train_ds.map(lambda x, y : (x, tf.one_hot(y, depth = 2)))
val_ds = val_ds.map(lambda x, y : (x, tf.one_hot(y,depth = 2)))


# In[45]:


lr = 0.000001      # The Learning Rate


# In[46]:


def alex_model(input_shape):
    
    input_img = tf.keras.Input(shape=input_shape)
    
    A1 = layer.Conv2D(filters=96, kernel_size=11, strides =(4,4), activation='ReLU')(input_img)
    P1 = layer.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')(A1)
    
    A2 = layer.Conv2D(filters=256, kernel_size=5, padding='same', activation='ReLU')(P1)
    P2 = layer.MaxPool2D(pool_size=(3,3), strides=(2,2))(A2)
    
    A3 = layer.Conv2D(filters=384, kernel_size=3, padding='same', activation='ReLU')(P2)
    
    A4 = layer.Conv2D(filters=384, kernel_size=3, padding='same', activation='ReLU')(A3)
    
    A5 = layer.Conv2D(filters=256, kernel_size=3, padding='same', activation='ReLU')(A4)
    P5 = layer.MaxPool2D(pool_size=(3,3), strides=(2,2))(A5)
    
    F = layer.Flatten()(P5)
    
    FC6 = layer.Dense(units=1024, activation='ReLU')(F)
    
    FC7 = layer.Dense(units=512, activation='ReLU')(FC6)
    
    outputs = layer.Dense(units = 2, activation = "softmax")(FC7)
    
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    
    return model


# In[47]:


Alex_model = alex_model((224, 224, 3))
Alex_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                   loss='binary_crossentropy',
                   metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, threshold=0.5), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
Alex_model.summary()


# In[48]:


checkpoint_path = 'training/conv/Alex_cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=False,
                                                save_freq ='epoch',
                                                verbose=1)
estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         min_delta=0.0001,
                                         patience=5,
                                         mode="min",
                                         restore_best_weights=True)


# In[50]:


historyA = Alex_model.fit(train_ds, 
                          epochs=10, 
                          validation_data=val_ds, 
                          callbacks = [cp_callback, estop])
Alex_model.save('AlexNet_model.h5')


# In[51]:


l = historyA.history.keys()
metrics = list(historyA.history.keys())
df = pd.DataFrame(historyA.history)
df.head()


# In[52]:


def f1_mod(x):
    return x[0]

def per_cent(x):
    return x*100
    
df['f1_score'] = df['f1_score'].apply(f1_mod)
df['val_f1_score'] = df['val_f1_score'].apply(f1_mod)

for i in df.columns:
    df[i] = df[i].apply(per_cent)


df.head()


# In[53]:


for i in range(len(l)//2):
    tr = metrics[i]
    val = "val_" + tr
    df_pl= df[[tr,val]]
    df_pl.rename(columns={tr:'Train',val:'Validation'},inplace=True)
    df_pl.plot(title='Model '+tr,figsize=(12,8)).set(xlabel='Epoch',ylabel=tr)


# In[54]:


def resnet_model(input_shape):
    
    input_img = tf.keras.Input(input_shape)
    
    base = tf.keras.applications.resnet50.ResNet50(input_shape = input_shape, weights = 'imagenet',
                                                   include_top = False, input_tensor = input_img)
    
    base.trainable = False
    
    A0 = base.output
    
    A1 = layer.GlobalAveragePooling2D( )(A0)
    N1 = layer.BatchNormalization()(A1)
    N1 = layer.Dropout(0.1)(N1)
    
    A2 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units=3072, activation='relu')(N1)
    N2 = layer.BatchNormalization()(A2)
    N2 = layer.Dropout(0.2)(N2)
    
    A3 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units=512, activation='relu')(N2)
    N3 = layer.BatchNormalization()(A3)
    N3 = layer.Dropout(0.4)(N3)
    
    outputs = layer.Dense(units = 2, activation = 'softmax')(N3)
    
    model = tf.keras.Model(inputs = input_img, outputs = outputs)
    
    return model


# In[55]:


checkpoint_path = 'training/conv/ResnetImgnet_cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=False,
                                                save_freq ='epoch',
                                                verbose=1)

estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         min_delta=0.0001,
                                         patience=5,
                                         mode="min",
                                         restore_best_weights=True)


# In[56]:


Resnet_model = resnet_model((224, 224, 3))
Resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                     loss='binary_crossentropy',
                     metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, threshold=0.5), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
Resnet_model.summary()


# In[ ]:


historyRI = Resnet_model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[cp_callback, estop])

Resnet_model.save('ResNet_model.h5')


# In[ ]:


l = historyRI.history.keys()
print(l, len(l), type(l))
metrics = list(historyRI.history.keys())

df = pd.DataFrame(historyRI.history)


# In[ ]:


def f1_mod(x):
    return x[0]

def per_cent(x):
    return x*100
    
df['f1_score'] = df['f1_score'].apply(f1_mod)
df['val_f1_score'] = df['val_f1_score'].apply(f1_mod)

for i in df.columns:
    df[i] = df[i].apply(per_cent)

df.head()


# In[ ]:


for i in range(len(l)//2):
    tr = metrics[i]
    val = "val_" + tr
    df_pl= df[[tr,val]]
    df_pl.rename(columns={tr:'train',val:'validation'},inplace=True)
    df_pl.plot(title='Model '+tr,figsize=(12,8)).set(xlabel='Epoch',ylabel=tr)


# In[ ]:


def vgg_model(input_shape, weights='imagenet',transfer=True):
    
    input_img = tf.keras.Input(shape=input_shape)
    
    base = tf.keras.applications.VGG16(input_shape = input_shape, weights = weights,
                                       include_top = False, input_tensor = input_img)
    base.trainable = not(transfer)
    
    A0 = base.output
    
    A1 = layer.GlobalAveragePooling2D( )(A0)
    N1 = layer.BatchNormalization()(A1)
    N1 = layer.Dropout(0.1)(N1)
    
    A2 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units = 256, activation = 'ReLU')(N1)
    N2 = layer.BatchNormalization()(A2)
    N2 = layer.Dropout(0.2)(N2)
    
    A3 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units = 128, activation = 'ReLU')(A2)
    N3 = layer.BatchNormalization()(A3)
    N3 = layer.Dropout(0.4)(N3)
    
    outputs = layer.Dense(units = 2, activation = "softmax")(N3)
    
    model = tf.keras.Model(inputs = input_img, outputs = outputs)
    
    return model


# In[ ]:


vggnet_I = vgg_model(input_shape=(224,224,3), weights='imagenet',transfer=True)

vggnet_I.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 loss='binary_crossentropy',
                 metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, threshold=0.5), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
vggnet_I.summary()


# In[ ]:


checkpoint_path = 'training/conv/VGG16Imgnet_cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=False,
                                                save_freq ='epoch',
                                                verbose=1)

estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         min_delta=0.0001,
                                         patience=5,
                                         mode="min",
                                         restore_best_weights=True)


# In[ ]:


historyVI = vggnet_I.fit(train_ds, 
                          epochs=10, 
                          validation_data=val_ds, 
                          callbacks = [cp_callback, estop])
vggnet_I.save('VGG16_model.h5')


# In[ ]:


l = historyVI.history.keys()
print(l, len(l), type(l))
metrics = list(historyVI.history.keys())

df = pd.DataFrame(historyVI.history)


# In[ ]:


def f1_mod(x):
    return x[0]

def per_cent(x):
    return x*100
    
df['f1_score'] = df['f1_score'].apply(f1_mod)
df['val_f1_score'] = df['val_f1_score'].apply(f1_mod)

for i in df.columns:
    df[i] = df[i].apply(per_cent)


df.head()


# In[ ]:


for i in range(len(l)//2):
    tr = metrics[i]
    val = "val_" + tr
    df_pl= df[[tr,val]]
    df_pl.rename(columns={tr:'train',val:'validation'},inplace=True)
    df_pl.plot(title='Model '+tr,figsize=(12,8)).set(xlabel='Epoch',ylabel=tr)


# In[ ]:


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import concatenate


# In[ ]:


def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 
  # Input: 
  # - f1: number of filters of the 1x1 convolutional layer in the first path
  # - f2_conv1, f2_conv3 are number of filters corresponding to the 1x1 and 3x3 convolutional layers in the second path
  # - f3_conv1, f3_conv5 are the number of filters corresponding to the 1x1 and 5x5  convolutional layer in the third path
  # - f4: number of filters of the 1x1 convolutional layer in the fourth path

  # 1st path:
  path1 = Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  # 2nd path
  path2 = Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = Conv2D(filters = f2_conv3, kernel_size = (3,3), padding = 'same', activation = 'relu')(path2)

  # 3rd path
  path3 = Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = Conv2D(filters = f3_conv5, kernel_size = (5,5), padding = 'same', activation = 'relu')(path3)

  # 4th path
  path4 = MaxPooling2D((3,3), strides= (1,1), padding = 'same')(input_layer)
  path4 = Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer


# In[ ]:


def GoogLeNet():
  # input layer 
  input_layer = Input(shape = (224, 224, 3))
  
  # convolutional layer: filters = 64, kernel_size = (7,7), strides = 2
  X = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'valid', activation = 'relu')(input_layer)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # convolutional layer: filters = 64, strides = 1
  X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)

  # convolutional layer: filters = 192, kernel_size = (3,3)
  X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

  # 1st Inception block
  X = Inception_block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)

  # 2nd Inception block
  X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

  # 3rd Inception block
  X = Inception_block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)

  # Extra network 1:
  X1 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
  X1 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
  X1 = Flatten()(X1)
  X1 = Dense(1024, activation = 'relu')(X1)
  X1 = Dropout(0.7)(X1)
  X1 = Dense(5, activation = 'softmax')(X1)

  
  # 4th Inception block
  X = Inception_block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)

  # 5th Inception block
  X = Inception_block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)

  # 6th Inception block
  X = Inception_block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)

  # Extra network 2:
  X2 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
  X2 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
  X2 = Flatten()(X2)
  X2 = Dense(1024, activation = 'relu')(X2)
  X2 = Dropout(0.7)(X2)
  X2 = Dense(1000, activation = 'softmax')(X2)
  
  
  # 7th Inception block
  X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, 
                      f3_conv5 = 128, f4 = 128)

  # max-pooling layer: pool_size = (3,3), strides = 2
  X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

  # 8th Inception block
  X = Inception_block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)

  # 9th Inception block
  X = Inception_block(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)

  # Global Average pooling layer 
  X = GlobalAveragePooling2D(name = 'GAPL')(X)

  # Dropoutlayer 
  X = Dropout(0.4)(X)

  # output layer 
  #X = Dense(1000, activation = 'softmax')(X)
  
  # model
  model = Model(input_layer, X, name = 'GoogLeNet')

  return model


# In[ ]:


def build_inception():
    
    base = GoogLeNet()
    
    A0 = base.output
    
    N1 = layer.BatchNormalization()(A0)
    
    A2 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units=3072, activation='relu')(N1)
    N2 = layer.BatchNormalization()(A2)
    N2 = layer.Dropout(0.2)(N2)
    
    A3 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units=512, activation='relu')(N2)
    N3 = layer.BatchNormalization()(A3)
    N3 = layer.Dropout(0.4)(N3)
    
    outputs = layer.Dense(units = 2, activation = "softmax")(N3)
    
    model = tf.keras.Model(inputs = base.input, outputs = outputs)
    
    return model    


# In[ ]:


Inc_model = build_inception()
Inc_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, threshold=0.5), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
Inc_model.summary()


# In[ ]:


checkpoint_path = 'training/conv/InceptionNet.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=False,
                                                save_freq ='epoch',
                                                verbose=1)

estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         min_delta=0.0001,
                                         patience=5,
                                         mode="min",
                                         restore_best_weights=True)


# In[ ]:


historyInc = Inc_model.fit(train_ds, 
                           epochs=10, 
                           validation_data=val_ds, 
                           callbacks = [cp_callback, estop])
Inc_model.save('InceptionNet_model.h5')


# In[ ]:


l = historyInc.history.keys()
print(l, len(l), type(l))
metrics = list(historyInc.history.keys())

df = pd.DataFrame(historyInc.history)


# In[ ]:


def f1_mod(x):
    return x[0]

def per_cent(x):
    return x*100
    
df['f1_score'] = df['f1_score'].apply(f1_mod)
df['val_f1_score'] = df['val_f1_score'].apply(f1_mod)

for i in df.columns:
    df[i] = df[i].apply(per_cent)


df.head()


# In[ ]:


for i in range(len(l)//2):
    tr = metrics[i]
    val = "val_" + tr
    df_pl= df[[tr,val]]
    df_pl.rename(columns={tr:'train',val:'validation'},inplace=True)
    df_pl.plot(title='Model '+tr,figsize=(12,8)).set(xlabel='Epoch',ylabel=tr)


# In[ ]:


def Mobilenet_model(input_shape, weights=None, transfer=False):
    
    input_img = tf.keras.Input(shape=input_shape)
    
    base = tf.keras.applications.MobileNetV2(input_shape = input_shape, weights = weights, 
                                             include_top = False, input_tensor = input_img)
    base.trainable = not(transfer)
    
    A0 = base.output
    
    A1 = layer.GlobalAveragePooling2D( )(A0)
    N1 = layer.BatchNormalization()(A1)
    N1 = layer.Dropout(0.1)(N1)
    
    A2 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units = 256, activation = 'ReLU')(N1)
    N2 = layer.BatchNormalization()(A2)
    N2 = layer.Dropout(0.2)(N2)
    
    A3 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units = 128, activation = 'ReLU')(A2)
    N3 = layer.BatchNormalization()(A3)
    N3 = layer.Dropout(0.4)(N3)
    
    outputs = layer.Dense(units = 2, activation = "softmax")(N3)
    
    model = tf.keras.Model(inputs = input_img, outputs = outputs)
    
    return model


# In[ ]:


mobile_I = Mobilenet_model(input_shape=(224,224,3), weights='imagenet',transfer=True)

mobile_I.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 loss='binary_crossentropy',
                 metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, threshold=0.5), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
                 
mobile_I.summary()


# In[ ]:


checkpoint_path = 'training/conv/Mobile_image_cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=False,
                                                save_freq ='epoch',
                                                verbose=1)

estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         min_delta=0.0001,
                                         patience=5,
                                         mode="min",
                                         restore_best_weights=True)


# In[ ]:


historyMi = mobile_I.fit(train_ds, 
                          epochs=10, 
                          validation_data=val_ds, 
                          callbacks = [cp_callback, estop])
mobile_I.save('MobileNet_model.h5')


# In[ ]:


l = historyMi.history.keys()
print(l, len(l), type(l))
metrics = list(historyMi.history.keys())

df = pd.DataFrame(historyMi.history)


# In[ ]:


def f1_mod(x):
    return x[0]

def per_cent(x):
    return x*100
    
df['f1_score'] = df['f1_score'].apply(f1_mod)
df['val_f1_score'] = df['val_f1_score'].apply(f1_mod)

for i in df.columns:
    df[i] = df[i].apply(per_cent)


df.head()


# In[ ]:


for i in range(len(l)//2):
    tr = metrics[i]
    val = "val_" + tr
    df_pl= df[[tr,val]]
    df_pl.rename(columns={tr:'train',val:'validation'},inplace=True)
    df_pl.plot(title='Model '+tr,figsize=(12,8)).set(xlabel='Epoch',ylabel=tr)


# In[ ]:


from tensorflow.keras import layers
from tensorflow import keras


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
img_size = (224, 224)
UNet_model = get_model(img_size, 2)
UNet_model.summary()


# In[ ]:


def Unet_classifier(base):
    
    A0 = base.output
    
    A1 = layer.GlobalAveragePooling2D( )(A0)
    N1 = layer.BatchNormalization()(A1)
    N1 = layer.Dropout(0.1)(N1)
    
    A2 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units=3072, activation='relu')(N1)
    N2 = layer.BatchNormalization()(A2)
    N2 = layer.Dropout(0.2)(N2)
    
    A3 = layer.Dense(kernel_regularizer=tf.keras.regularizers.L2(0.0002),
                     units=512, activation='relu')(N2)
    N3 = layer.BatchNormalization()(A3)
    N3 = layer.Dropout(0.4)(N3)
    
    outputs = layer.Dense(units = 2, activation = "softmax")(N3)
    
    model = tf.keras.Model(inputs = base.input, outputs = outputs)
    
    return model


# In[ ]:


Unet = Unet_classifier(UNet_model)
Unet.summary()


# In[ ]:


Unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 loss='binary_crossentropy',
                 metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, threshold=0.5), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


# In[ ]:


checkpoint_path = 'training/conv/UNet1.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=False,
                                                save_freq ='epoch',
                                                verbose=1)

estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         min_delta=0.0001,
                                         patience=5,
                                         mode="min",
                                         restore_best_weights=True)


# In[ ]:


historyU = Unet.fit(train_ds, epochs=10, validation_data=val_ds, callbacks = [cp_callback, estop])
Unet.save('UNet.h5')


# In[ ]:


l = historyU.history.keys()
print(l, len(l), type(l))
metrics = list(historyU.history.keys())

df = pd.DataFrame(historyU.history)


# In[ ]:


def f1_mod(x):
    return x[0]

def per_cent(x):
    return x*100
    
df['f1_score'] = df['f1_score'].apply(f1_mod)
df['val_f1_score'] = df['val_f1_score'].apply(f1_mod)

for i in df.columns:
    df[i] = df[i].apply(per_cent)


df.head()


# In[ ]:


for i in range(len(l)//2):
    tr = metrics[i]
    val = "val_" + tr
    df_pl= df[[tr,val]]
    df_pl.rename(columns={tr:'train',val:'validation'},inplace=True)
    df_pl.plot(title='Model '+tr,figsize=(12,8)).set(xlabel='Epoch',ylabel=tr)


# In[ ]:





# In[ ]:





# In[ ]:




