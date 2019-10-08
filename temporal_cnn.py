from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Dense, InputLayer, Flatten
from helpers.read_json import return_joint_cartesian
import numpy as np

json_file = open("JSONs/Walking/walking_0.json")
joint_cartesian = return_joint_cartesian(json_file)

lookahead_size = 100
X = []
y = []
for index in range(len(joint_cartesian) - (lookahead_size + 1)):
    X.append(joint_cartesian[index:index + lookahead_size])
    y.append(joint_cartesian[index + (lookahead_size + 1)])

X = np.array(X)
y = np.array(y)
y = np.reshape(y, (-1, 78))
print(f"Train data shape: {X.shape}")
print(f"Test data shape: {y.shape}")

temporal_model = Sequential()
temporal_model.add(InputLayer(input_shape=(lookahead_size, 26, 3)))
temporal_model.add(Conv2D(32, (7, 5), activation="relu"))
temporal_model.add(Conv2D(32, (7, 3), activation="relu"))
temporal_model.add(MaxPool2D(2))
temporal_model.add(Conv2D(64, (7, 3), activation="relu"))
temporal_model.add(Conv2D(64, (7, 3), activation="relu"))
temporal_model.add(MaxPool2D(2))
temporal_model.add(Conv2D(128, (7, 3), activation="relu"))
temporal_model.add(Flatten())
temporal_model.add(Dense(1280, activation="relu"))
temporal_model.add(Dense(1280, activation="relu"))
temporal_model.add(Dense(78, activation="linear"))
temporal_model.compile(optimizer="adam", loss="mse")
temporal_model.summary()

temporal_model.fit(X, y, batch_size=16, epochs=1000)

