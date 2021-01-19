# based on de-noising auto-encoder found in Keras

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Lambda, concatenate
from keras.models import Model
from random import randint
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

training_file = open('CSVs/training.csv', 'r')
training_file2 = open('CSVs/training2.csv', 'r')
target_file = open('CSVs/target.csv')
training_data = []
training_data2 = []
target_data = []

v_training_file = open('NW-CSVs/training-2-1.csv', 'r')
v_training_file2 = open('NW-CSVs/training-2-2.csv', 'r')
v_target_file = open('NW-CSVs/target-2.csv')
v_training_data = []
v_training_data2 = []
v_target_data = []

'''
Training data
'''
for line in training_file.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                training_data.append(float(data))
            except ValueError:
                pass

for line in training_file2.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                training_data2.append(float(data))
            except ValueError:
                pass

for line in target_file.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                target_data.append(float(data))
            except ValueError:
                pass

'''
Validation data
'''
for line in v_training_file.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                v_training_data.append(float(data))
            except ValueError:
                pass

for line in v_training_file2.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                v_training_data2.append(float(data))
            except ValueError:
                pass

for line in v_target_file.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                v_target_data.append(float(data))
            except ValueError:
                pass




# Setting both as target data just for testing purposes to get past errors
x_train = np.reshape(training_data, (-1, 9, 1))  # adapt this if using `channels_first` image data format
x_train2 = np.reshape(training_data2, (-1, 9, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(target_data, (-1, 18, 1))  # adapt this if using `channels_first` image data format

x_train = x_train.astype('float32')
x_train2 = x_train2.astype('float32')
x_test = x_test.astype('float32')

v_train = np.reshape(v_training_data, (-1, 9, 1))  # adapt this if using `channels_first` image data format
v_train2 = np.reshape(v_training_data2, (-1, 9, 1))  # adapt this if using `channels_first` image data format
v_test = np.reshape(v_target_data, (-1, 18, 1))  # adapt this if using `channels_first` image data format

v_train = v_train.astype('float32')
v_train2 = v_train2.astype('float32')
v_test = v_test.astype('float32')


# https://stackoverflow.com/questions/43196636/how-to-concatenate-two-layers-in-keras
input_img = Input(shape=(9, 1))  # adapt this if using `channels_first` image data format
x = Conv1D(32, 3, activation='linear', padding='same')(input_img)
x = Dropout(0.5)(x)
encoded = MaxPooling1D(3, padding='same')(x)

x = Conv1D(32, 3, activation='linear', padding='same')(x)
# x = Dropout(0.2)(x)
x = UpSampling1D(1)(x)
decoded = Conv1D(1, 3, activation='linear', padding='same')(x)

input_img2 = Input(shape=(9, 1))  # adapt this if using `channels_first` image data format
x2 = Conv1D(32, 3, activation='linear', padding='same')(input_img2)
x = Dropout(0.5)(x)
encoded2 = MaxPooling1D(3, padding='same')(x2)

x2 = Conv1D(32, 3, activation='linear', padding='same')(x2)
# x = Dropout(0.2)(x)
x2 = UpSampling1D(1)(x2)
decoded2 = Conv1D(1, 3, activation='linear', padding='same')(x2)
concat = concatenate([decoded, decoded2], axis=1)

autoencoder = Model(inputs=[input_img, input_img2], outputs=concat)
autoencoder.compile("adam", loss='mse')

autoencoder.fit([x_train, x_train2], x_test,
                epochs=1000,
                batch_size=128,
                shuffle=True,
                # validation_data=([v_train, v_train2], v_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

autoencoder.save("Models/dae-xyz-v3n-crazy.h5")