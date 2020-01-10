from networks.wgan.wgan import WGAN
from sklearn.preprocessing import MinMaxScaler
from glob import glob
import numpy as np

data = glob("data/*.csv")
X_train = None
scaler = MinMaxScaler
joint_scalers = []
batch_size = 1

readlines_arr = []
for file in data:
    file = open(file)
    readlines_arr += file.readlines()[:500]
    file.close()

X_train = []
for line in readlines_arr:
    line = line.replace("\n", "").split(",")[:-1]
    try:
        line = [float(value) for value in line][:72]
        if len(line) > 3:
            X_train.append(line)
    except ValueError:
        continue

X_train = np.array(X_train)
X = np.reshape(X_train, (-1, 72))
X = np.transpose(X)

new_X = []
for joint_data in X:
    this_joint_mms = MinMaxScaler((0, 1), False)
    joint_data = np.reshape(joint_data, (-1, 1))
    this_joint_mms.fit_transform(joint_data)
    joint_scalers.append(this_joint_mms)
    new_X.append(joint_data)
X = np.array(new_X)
X = np.reshape(X, (72, -1))
X = np.transpose(X)

X = np.reshape(X, (-1, 100, 24, 3))
print(X.shape)

wgan = WGAN(X.shape[1], X.shape[2], joint_scalers=joint_scalers)
wgan.train(X_train=X, epochs=4000, batch_size=16, sample_interval=50)
