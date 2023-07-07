import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model

X_file = '전처리 npy 데이터/X.new.npy'
Y_file = '전처리 npy 데이터/Y.new.npy'


# 파일 불러오기
X = np.load(X_file)
Y = np.load(Y_file)


# 입력 데이터의 크기 지정
img_width, img_height = 64,64

# 체크포인트 경로 및 파일 이름 지정
filepath="학습모델/saved_model.h5"

# 모델 아키텍처 정의
input_shape = (50, img_width, img_height, 3)

inputs = Input(shape=input_shape)

# CNN 블록
x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2)))(x)
x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2)))(x)
x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), strides=(2, 2)))(x)
x = layers.TimeDistributed(layers.Flatten())(x)

# RNN 블록 
x = layers.GRU(128, return_sequences=True)(x)
x = layers.GRU(128, return_sequences=True)(x)
x = layers.GRU(128, return_sequences=True)(x) 

# DNN 블록
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# 학습
tf.config.run_functions_eagerly(True)
checkpoint = ModelCheckpoint(filepath, save_weights_only=False ,monitor='val_loss', verbose=1, save_best_only=False, mode='min')

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'],run_eagerly=True)

result = model.fit(X,Y, epochs=15 ,batch_size = 1,callbacks=[checkpoint])
plot_model(model,show_shapes=False,to_file='model.png')

