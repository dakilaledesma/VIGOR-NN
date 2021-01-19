import json
from math import radians, cos


def convert(read, write):
    array = open(read + '.json')
    save_file = open(write + '.csv', 'w')
    userdata = json.load(array)

    for frame in range(0, 5000):
        for joint in [0, 1, 2, 3, 4, 5]:
            save_file.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['x'])
            )
            save_file.write(",")
            save_file.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['y'])
            )
            save_file.write(",")
            save_file.write(
                str(userdata["ModelQuaternionList"][frame]['V3BSVJoints'][int(joint)]['z'])
            )
            save_file.write(",")
        save_file.write("\n")
    save_file.close()

convert('../JSONs/relative-natural-3', 'CSVs/training-gta')
