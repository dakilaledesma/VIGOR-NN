from keras.models import load_model
import numpy as np
from math import acos, degrees

autoencoder = load_model("Models/dae-xyz-v4b.h5")
training_file = open('CSVs/training-vid.csv', 'r')
training_file2 = open('CSVs/training-vid2.csv', 'r')

training_data = []
training_data2 = []

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
training_data = np.reshape(training_data, (-1, 9, 1))
training_data2 = np.reshape(training_data2, (-1, 9, 1))
prediction_data = autoencoder.predict([training_data, training_data2])

prediction_file = open('CSVs/prediction-vidsmall.csv', 'w')
prediction_data = np.reshape(prediction_data, (-1, 18, 1))

for frame in prediction_data:
    for data in frame:
        prediction_file.write(str(data[0]))
        prediction_file.write(",")
    prediction_file.write("\n")



