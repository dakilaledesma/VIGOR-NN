import numpy as np
from math import acos, degrees

gt = open('CSVs/target.csv', 'r')
predfile = open('CSVs/prediction.csv', 'r')

gtdata = []
preddata = []

for line in gt.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                gtdata.append(float(data))
            except ValueError:
                pass

for line in predfile.readlines():
    for data in line.split(","):
        if data != "\n" or data != "":
            try:
                preddata.append(float(data))
            except ValueError:
                pass

gtdata = np.reshape(gtdata, (-1, 18, 1))
preddata = np.reshape(preddata, (-1, 18, 1))

distance = np.linalg.norm(gtdata-preddata)
print(distance)
